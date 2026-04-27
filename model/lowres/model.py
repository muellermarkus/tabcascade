import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm

from .layers import CatEmbedding, Timewarp, Timewarp_Logistic, WeightNetwork
from .utils import low_discrepancy_sampler, set_seeds


class CatCDTD(nn.Module):
    """
    This implements the categorical part of the CDTD.
    """

    def __init__(
        self,
        score_model,
        num_classes,
        proportions,
        emb_dim,
        sigma_min=1e-5,
        sigma_max=100,
        sigma_data=1,
        normalize_by_entropy=True,
        weight_low_noise=1.0,
        timewarp_variant="logistic",
        cat_emb_init_sigma=0.001,
    ):
        super().__init__()
        self.num_features = len(num_classes)
        self.num_classes = num_classes
        self.sigma_data = sigma_data
        self.emb_dim = emb_dim

        if normalize_by_entropy:
            entropy = torch.tensor([-torch.sum(p * p.log()) for p in proportions])
        else:
            entropy = torch.ones(self.num_features)
        self.register_buffer("entropy", entropy)

        self.score_model = score_model
        self.weightnet = WeightNetwork(1024)
        self.encoder = CatEmbedding(emb_dim, num_classes, cat_emb_init_sigma, bias=True, normalize_emb=True)

        self.timewarp_variant = timewarp_variant
        if timewarp_variant == "logistic":
            self.timewarp = Timewarp_Logistic(
                "single",
                self.num_features,
                0,
                torch.tensor(sigma_min),
                torch.tensor(sigma_max),
                weight_low_noise=weight_low_noise,
                decay=0,
            )
        elif timewarp_variant == "pwl":
            self.timewarp = Timewarp(sigma_min=sigma_min, sigma_max=sigma_max, decay=0.1)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def loss_fn(self, x, t=None, validation=False):
        B = x.shape[0]
        x_emb = self.encoder(x)

        # draw time and convert to standard deviations for noise
        with torch.no_grad():
            if t is None:
                t = low_discrepancy_sampler(B, device=self.device)  # (B,)

            if not validation:
                sigma = self.timewarp.get_sigmas(t)
            else:
                sigma = (
                    low_discrepancy_sampler(B, device=self.device).to(torch.float32)
                    * (self.timewarp.sigma_max - self.timewarp.sigma_min)
                    + self.timewarp.sigma_min
                )
                t = self.timewarp.get_t(sigma)
            sigma = repeat(sigma, "B F -> B F G", F=self.num_features, G=1)

            t = t.to(torch.float32)
            assert sigma.shape == (B, self.num_features, 1)

        # add noise
        x_emb_t = x_emb + torch.randn_like(x_emb) * sigma

        # pass to score model
        logits = self.precondition(x_emb_t, t, sigma)

        ce_losses = torch.stack(
            [F.cross_entropy(logits[i], x[:, i], reduction="none") for i in range(self.num_features)],
            dim=1,
        )

        losses = {}
        losses["weighted"] = ce_losses / (self.entropy + 1e-8)
        time_reweight = self.weightnet(t).unsqueeze(1)
        losses["weighted_calibrated"] = losses["weighted"] / time_reweight.exp().detach()

        if self.timewarp_variant == "logistic":
            losses["timewarping"] = self.timewarp.loss_fn(sigma.squeeze(-1).detach(), losses["weighted"].detach())
        elif self.timewarp_variant == "pwl":
            losses["timewarping"] = self.timewarp.loss_fn(
                sigma.detach()[:, 0].squeeze(-1), losses["weighted"].detach().mean(1).squeeze(-1)
            )

        losses["weightnet"] = (time_reweight.exp() - losses["weighted"].detach().mean(1)) ** 2

        train_loss = losses["weighted_calibrated"].mean() + losses["timewarping"].mean() + losses["weightnet"].mean()

        losses["train_loss"] = train_loss

        return losses

    def precondition(self, x_emb_t, t, sigma):
        """
        Improved preconditioning proposed in the paper "Elucidating the Design
        Space of Diffusion-Based Generative Models" (EDM) adjusted for categorical data
        """

        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()  # (B, num_features, 1)
        c_noise = torch.log(t + 1e-8) * 0.25

        return self.score_model(c_in * x_emb_t, c_noise * 1000)

    def init_score_interpolation(self):
        # copy data to embedding bag
        full_emb = self.encoder.cat_emb.weight.data.detach()

        # add bias to embedding bag
        if self.encoder.bias:
            bias = []
            for i in range(self.num_features):
                bias.append(self.encoder.cat_bias[i].unsqueeze(0).expand(self.num_classes[i], -1))
            bias = torch.row_stack(bias)

        assert bias.shape == full_emb.shape
        full_emb = full_emb + bias

        # before running score interpolation, normalize embedding bag weights once
        full_emb = F.normalize(full_emb, dim=1, eps=1e-20) * torch.tensor(self.emb_dim).sqrt()
        full_emb = full_emb.to(torch.float64)

        self.embeddings_score_interp = torch.split(full_emb, self.num_classes, dim=0)

    def score_interpolation(self, x_emb_t, logits, sigma, return_probs=False):
        if return_probs:
            # transform logits for categorical features to probabilities
            probs = [F.softmax(l.to(torch.float64), dim=1) for l in logits]
            return probs

        x_emb_hat = torch.zeros_like(x_emb_t, device=self.device, dtype=torch.float64)

        for i, logs in enumerate(logits):
            probs = F.softmax(logs.to(torch.float64), dim=1)
            x_emb_hat[:, i, :] = torch.matmul(probs, self.embeddings_score_interp[i])

        # plug interpolated embedding into score function to interpolate score
        interpolated_score = (x_emb_t - x_emb_hat) / sigma

        return interpolated_score, x_emb_hat

    @torch.inference_mode()
    def sampler(self, latents, num_steps=200):
        B = latents.shape[0]

        # construct time steps
        t_steps = torch.linspace(1, 0, num_steps + 1, device=self.device, dtype=torch.float64)
        s_steps = self.timewarp.get_sigmas(t_steps).to(torch.float64)

        assert torch.allclose(s_steps[0].to(torch.float32), self.timewarp.sigma_max.float())
        assert torch.allclose(s_steps[-1].to(torch.float32), self.timewarp.sigma_min.float())
        # the final step goes onto t = 0, i.e., sigma = sigma_min = 0

        # initialize latents at maximum noise level
        x_next = latents.to(torch.float64) * s_steps[0].unsqueeze(1)

        for i, (s_cur, s_next, t_cur) in enumerate(zip(s_steps[:-1], s_steps[1:], t_steps[:-1])):
            s_cur = s_cur.repeat((B, 1))
            s_next = s_next.repeat((B, 1))

            # get score model output
            logits = self.precondition(
                x_emb_t=x_next.to(torch.float32),
                t=t_cur.to(torch.float32).repeat((B,)),
                sigma=s_cur.to(torch.float32).unsqueeze(-1),
            )

            # estimate scores
            d_cur, _ = self.score_interpolation(x_next, logits, s_cur.unsqueeze(-1))

            # adjust data samples
            h = s_next - s_cur
            x_next = x_next + h.unsqueeze(-1) * d_cur

        # final prediction of classes for categorical feature
        t_final = t_steps[:-1][-1]
        s_final = s_steps[:-1][-1].repeat(B, 1)

        logits = self.precondition(
            x_emb_t=x_next.to(torch.float32),
            t=t_final.to(torch.float32).repeat((B,)),
            sigma=s_final.to(torch.float32).unsqueeze(-1),
        )

        # get probabilities for each category and derive generated classes
        probs = self.score_interpolation(x_next, logits, s_final.unsqueeze(-1), return_probs=True)
        x_gen = torch.empty(B, self.num_features, device=self.device)
        for i in range(self.num_features):
            x_gen[:, i] = probs[i].argmax(1)

        return x_gen.cpu()

    @torch.inference_mode()
    def sample_data(self, num_samples, num_steps=200, batch_size=4096, seed=42, verbose=True):
        # init required data
        self.init_score_interpolation()

        set_seeds(seed, cuda_deterministic=True)
        n_batches, remainder = divmod(num_samples, batch_size)
        sample_sizes = n_batches * [batch_size] + [remainder] if remainder != 0 else n_batches * [batch_size]

        x = []
        for i, num_samples in enumerate(tqdm(sample_sizes, disable=(not verbose))):
            latents = torch.randn((num_samples, self.num_features, self.emb_dim), device=self.device)
            x_gen = self.sampler(latents, num_steps=num_steps)
            x.append(x_gen)

        x = torch.cat(x).cpu().long()

        return x

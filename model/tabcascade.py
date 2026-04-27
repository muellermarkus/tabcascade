from pathlib import Path

import torch
from sklearn.preprocessing import OrdinalEncoder
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from model.highres.model import HighResFlowModel
from model.lowres.encoder import Discretizer
from model.lowres.layers import LowResMLP
from model.lowres.model import CatCDTD
from model.lowres.utils import FastTensorDataLoader, cycle


class TabCascade:
    def __init__(self, config, seed=0):
        super().__init__()
        self.seed = seed
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode_into_z(self, x_num):

        self.z_encoder = Discretizer(
            x_num,
            variant=self.config.data.encoder,
            seed=self.seed,
            k_max=self.config.data.k_max,
            max_depth=self.config.data.max_depth,
        )
        groups, mask, infl_groups, has_miss = self.z_encoder.encode(x_num)

        if self.config.data.encoder == "gmm":
            # adjust means and remove those not appearing in the data (after hard clustering)
            for i in range(groups.shape[1]):
                vals = groups[:, i].unique()
                self.z_encoder.means[i] = self.z_encoder.means[i][vals]

            # train additional ordinal encoder for groups
            # as some components may never be the argmax and thus not appear in the data
            self.gmm_ord_enc = OrdinalEncoder()
            groups = self.gmm_ord_enc.fit_transform(groups.numpy())
            groups = torch.from_numpy(groups).long()

        # provide easy access to group-specific means and standard deviations
        self.z_means = self.z_encoder.means
        self.z_stds = self.z_encoder.stds

        return (
            groups,
            mask,
            infl_groups,
            has_miss,
        )

    def get_masks(self, groups: torch.Tensor):
        # gets masks for generated Z_num (from lowres model)

        # get inflated mask
        infl_mask = []
        for i in range(groups.shape[1]):
            # account for shift in groups if there are missings (then missing group = 0)
            z_infl_groups = torch.tensor(self.z_infl_groups[i])
            infl_groups = z_infl_groups + 1 if self.z_has_miss[i] else z_infl_groups
            mask = torch.isin(groups[:, i], infl_groups)
            infl_mask.append(mask)
        infl_mask = torch.column_stack(infl_mask)

        # get missingness mask
        miss_mask = []
        for i in range(groups.shape[1]):
            if self.z_has_miss[i]:
                miss_mask.append(groups[:, i] == 0)
            else:
                miss_mask.append(torch.zeros_like(groups[:, i]).bool())
        miss_mask = torch.column_stack(miss_mask) if self.z_has_miss.any() else None

        return infl_mask, miss_mask

    def get_classes_and_proportions(self, x_cat, groups):

        # determine n_classes for X_cat
        n_classes_cat = []
        proportions_cat = []
        n_sample = x_cat.shape[0]
        for i in range(x_cat.shape[1]):
            val, counts = x_cat[:, i].unique(return_counts=True)
            n_classes_cat.append(len(val))
            proportions_cat.append(counts / n_sample)

        # determine n_classes for Z_num
        n_classes_num = []
        proportions_num = []
        for i in range(groups.shape[1]):
            val, counts = groups[:, i].unique(return_counts=True)
            n_classes_num.append(len(val))
            proportions_num.append(counts / n_sample)

        # concate n_classes and proportions for x_low = (X_cat, Z_num)
        n_classes = n_classes_cat + n_classes_num
        proportions = proportions_cat + proportions_num

        return n_classes, proportions

    def get_train_loader(self, x_cat, x_num, z_groups, z_mask):

        # mean impute numerical values
        x_means = torch.nanmean(x_num, dim=0)
        for i in range(x_num.shape[1]):
            x_num[:, i] = torch.nan_to_num(x_num[:, i], nan=x_means[i])

        batch_size = min(self.config.data.batch_size, x_num.shape[0])
        train_loader = FastTensorDataLoader(
            x_cat,
            x_num,
            z_groups,
            z_mask,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        return train_loader

    def get_lowres_model(self):
        cfg = self.config.lowres.model
        predictor = LowResMLP(
            self.n_classes,
            cfg.cat_emb_dim,
            cfg.mlp_emb_dim,
            cfg.mlp_n_layers,
            cfg.mlp_n_units,
            self.proportions,
            cfg.mlp_act,
        )
        lowres = CatCDTD(
            predictor,
            self.n_classes,
            self.proportions,
            cfg.cat_emb_dim,
            cfg.sigma_min,
            cfg.sigma_max,
            cfg.sigma_data,
            cfg.normalize_by_entropy,
            cfg.timewarp_weight_low_noise,
            cfg.timewarp_variant,
            cfg.cat_emb_init_sigma,
        )
        return lowres

    def get_highres_model(self):
        highres = HighResFlowModel(
            self.z_means,
            self.z_stds,
            self.n_classes,
            self.config.highres.model.mlp_emb_dim,
            self.config.highres.model.mlp_n_layers,
            self.config.highres.model.mlp_n_units,
            self.config.highres.model.gamma_input_dim,
            self.config.highres.model.cat_emb_dim,
        )
        return highres

    def train(self, x_cat, x_num):

        self.n_cat_cols = x_cat.shape[1]
        z_groups, z_mask, self.z_infl_groups, self.z_has_miss = self.encode_into_z(x_num)
        self.n_classes, self.proportions = self.get_classes_and_proportions(x_cat, z_groups)
        self.train_loader = self.get_train_loader(x_cat, x_num, z_groups, z_mask)
        self.lowres = self.get_lowres_model().to(self.device)
        self.highres = self.get_highres_model().to(self.device)

        num_params_lowres = sum(p.numel() for p in self.lowres.parameters())
        num_params_highres = sum(p.numel() for p in self.highres.parameters())
        print("Total parameters =", num_params_highres + num_params_lowres)
        print("Lowres parameters =", num_params_lowres)
        print("Highres parameters =", num_params_highres)

        ema_lowres = ExponentialMovingAverage(
            self.lowres.parameters(),
            decay=self.config.lowres.training.ema_decay,
        )

        ema_highres = ExponentialMovingAverage(
            self.highres.parameters(),
            decay=self.config.highres.training.ema_decay,
        )

        opt_lowres = torch.optim.AdamW(
            self.lowres.parameters(),
            lr=self.config.lowres.training.lr,
            weight_decay=self.config.lowres.training.weight_decay,
            betas=self.config.lowres.training.betas,
        )

        opt_highres = torch.optim.AdamW(
            self.highres.parameters(),
            lr=self.config.highres.training.lr,
            weight_decay=self.config.highres.training.weight_decay,
            betas=self.config.highres.training.betas,
        )

        scheduler_highres = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_highres,
            mode="min",
            factor=0.9,
            patience=3,
            min_lr=1e-6,
        )

        train_loader = cycle(self.train_loader)
        step = 0
        n_inputs = 0
        lowres_loss_trn = 0
        lowres_loss_hist = []
        highres_loss_trn = 0
        highres_loss_hist = []

        pbar = tqdm(total=self.config.lowres.training.num_steps_train)

        while step < self.config.lowres.training.num_steps_train:
            # Linear warmup learning rate for lowres model
            if step < self.config.lowres.training.num_steps_warmup:
                lr = self.config.lowres.training.lr * (step + 1) / self.config.lowres.training.num_steps_warmup
                for param_group in opt_lowres.param_groups:
                    param_group["lr"] = lr

            # Linear decay for cdtd-based lowres model
            if (self.config.lowres.model.variant == "cdtd") and (step > self.config.lowres.training.num_steps_warmup):
                aux_step = step - self.config.lowres.training.num_steps_warmup
                rate = 1 - (
                    aux_step
                    / (self.config.lowres.training.num_steps_train - self.config.lowres.training.num_steps_warmup)
                )
                lr = self.config.lowres.training.lr * rate + 1e-6 * (1 - rate)
                for param_group in opt_lowres.param_groups:
                    param_group["lr"] = lr

            # Linear warmup learning rate for highres model
            if step < self.config.highres.training.num_steps_warmup:
                lr = self.config.highres.training.lr * (step + 1) / self.config.highres.training.num_steps_warmup
                for param_group in opt_highres.param_groups:
                    param_group["lr"] = lr

            # Linear decay for cdtd-based highres model
            if self.config.highres.model.get("variant", "flow") == "cdtd" and (
                step > self.config.highres.training.num_steps_warmup
            ):
                aux_step = step - self.config.highres.training.num_steps_warmup
                rate = 1 - (
                    aux_step
                    / (self.config.lowres.training.num_steps_train - self.config.highres.training.num_steps_warmup)
                )
                lr = self.config.highres.training.lr * rate + 1e-6 * (1 - rate)
                for param_group in opt_highres.param_groups:
                    param_group["lr"] = lr

            opt_lowres.zero_grad(set_to_none=True)
            opt_highres.zero_grad(set_to_none=True)

            batch = next(train_loader)
            x_cat, x_num, z_num, mask = (x.to(self.device) for x in batch)
            B = len(x_cat)
            n_inputs += B

            ################################
            # lowres model
            lowres_input = torch.column_stack((x_cat, z_num))
            losses = self.lowres.loss_fn(lowres_input)
            train_loss_lowres = losses["train_loss"]
            train_loss_lowres.backward()

            if self.config.lowres.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.lowres.parameters(), max_norm=1.0)
            opt_lowres.step()
            ema_lowres.update()
            lowres_loss_trn += train_loss_lowres.detach().item() * B

            ################################
            # highres model
            train_loss_highres = self.highres.loss_fn(x_num, x_cat, z_num, mask)
            train_loss_highres.backward()

            if self.config.highres.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.highres.parameters(), max_norm=1.0)
            opt_highres.step()
            ema_highres.update()
            highres_loss_trn += train_loss_highres.detach().item() * B

            ################################
            # bookkeeping and learning rate scheduling

            if step % self.config.lowres.training.log_steps == 0:
                lowres_loss_trn = lowres_loss_trn / n_inputs
                lowres_loss_hist.append(lowres_loss_trn)
                highres_loss_trn = highres_loss_trn / n_inputs
                highres_loss_hist.append(highres_loss_trn)
                pbar.set_postfix(
                    {"loss (lowres)": f"{lowres_loss_trn:.4f}", "loss (highres)": f"{highres_loss_trn:.4f}"},
                )
                lowres_loss_trn = 0
                highres_loss_trn = 0
                n_inputs = 0
                scheduler_highres.step(highres_loss_trn)
            step += 1
            pbar.update(1)
        pbar.close()

        # copy EMA weights to the model
        ema_lowres.copy_to()
        self.lowres.eval()
        ema_highres.copy_to()
        self.highres.eval()

    def save_model(self, save_dir: Path):
        state = {
            "n_cat_cols": self.n_cat_cols,
            "n_classes": self.n_classes,
            "proportions": self.proportions,
            "z_means": self.z_means,
            "z_stds": self.z_stds,
            "z_infl_groups": self.z_infl_groups,
            "z_has_miss": self.z_has_miss,
            "lowres": self.lowres.state_dict(),
            "highres": self.highres.state_dict(),
        }
        torch.save(state, save_dir / "model.pt")

        if self.config.data.encoder == "gmm":
            torch.save(self.gmm_ord_enc, save_dir / "gmm_ord_enc.pt")

    def load_model(self, save_dir: Path):

        state = torch.load(save_dir / "model.pt", weights_only=False)
        self.n_cat_cols = state["n_cat_cols"]
        self.n_classes = state["n_classes"]
        self.proportions = state["proportions"]
        self.z_means = state["z_means"]
        self.z_stds = state["z_stds"]
        self.z_infl_groups = state["z_infl_groups"]
        self.z_has_miss = state["z_has_miss"]

        self.lowres = self.get_lowres_model()
        self.highres = self.get_highres_model()
        self.lowres.load_state_dict(state["lowres"])
        self.highres.load_state_dict(state["highres"])
        self.lowres.to(self.device).eval()
        self.highres.to(self.device).eval()

    def sample(self, num_samples):

        # first sample x_low = (x_cat, z_num)
        x_low_gen = self.lowres.sample_data(
            num_samples,
            num_steps=self.config.lowres.model.generation_steps,
            batch_size=self.config.lowres.model.generation_batch_size,
            seed=self.seed,
            verbose=False,
        )
        x_cat_gen = x_low_gen[:, : self.n_cat_cols]
        z_num_gen = x_low_gen[:, self.n_cat_cols :]

        # then sample high resolution information conditioned on low resolution
        x_num_gen = self.highres.sample_data(
            x_cat_gen,
            z_num_gen,
            num_steps=self.config.highres.model.generation_steps,
            batch_size=self.config.highres.model.generation_batch_size,
            seed=self.seed,
            verbose=False,
        )

        # overwrite inflated / missing values in X_num using Z_num
        assert x_num_gen.shape == z_num_gen.shape
        if self.config.data.encoder == "gmm":
            z_num_gen_enc = self.gmm_ord_enc.inverse_transform(z_num_gen)
            z_num_gen_enc = torch.from_numpy(z_num_gen_enc).long()
        else:
            z_num_gen_enc = z_num_gen
        infl_mask, miss_mask = self.get_masks(z_num_gen_enc)

        # get groups means (= inflated value if var = 0)
        z_num_gen_means = (
            self.highres.get_group_means(z_num_gen.to(self.device) + self.highres.group_offset).squeeze(-1).cpu()
        )
        x_num_gen = torch.where(infl_mask, z_num_gen_means, x_num_gen)

        if miss_mask is not None:
            x_num_gen = torch.masked_fill(x_num_gen, miss_mask, torch.nan)

        return x_cat_gen.numpy(), x_num_gen.numpy()

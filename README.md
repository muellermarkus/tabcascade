# Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features
<p align="center">
  <a href="https://arxiv.org/abs/2601.22816">
    <img alt="Paper URL" src="https://img.shields.io/badge/cs.LG-2601.22816-B31B1B.svg">
  </a>
</p>

Our paper introduces **TabCascade**, a cascaded flow matching model for high-fidelity tabular data generation. On average, TabCascade improves the detection score by over 50% compared to the baselines. If needed, TabCascade is able to generate realistic missing values in numerical features.

![Banner](https://github.com/muellermarkus/tabcascade/blob/master/figures/overview.png)

## Abstract

Advances in generative modeling have recently been adapted to tabular data containing discrete and continuous features.
However, generating mixed-type features that combine discrete states with an otherwise continuous distribution in a single feature remains challenging.
We advance the state-of-the-art in diffusion models for tabular data with a cascaded approach. We first generate a low-resolution version of a tabular data row, that is, the collection of the purely categorical features and a coarse categorical representation of numerical features. Next, this information is leveraged in the high-resolution flow matching model via a novel guided conditional probability path and data-dependent coupling. The low-resolution representation of numerical features explicitly accounts for discrete outcomes, such as missing or inflated values, and therewith enables a more faithful generation of mixed-type features.
We formally prove that this cascade tightens the transport cost bound. The results indicate that our model generates significantly more realistic samples and captures distributional details more accurately, for example, the detection score improves by 51.9\%.

## Install instructions
Our environment depends on `rpy2`, hence we require `R` to be installed on the system. Otherwise `uv sync` will fail. Make sure uv is installed and run `uv sync`.

### Installing disttree

Our encoding of $\mathbf{x}_{\text{num}}$ into $\mathbf{z}$ requires the disttree R package for distributional regressiona trees.
With `rpy2` installed, simply `uv run disttree/model.py` to install the necessary packages. Note that the installation of `rpy2` requires `R` to be installed on your system. Note that for some systems custom steps may be needed dependent on your R installation and its interaction with `rpy2`.
If installation completes successfully, a matrix of groups retrieved for some simulated $\mathbf{x}_{\text{num}}$ should be displayed to confirm that disttree works as intended.

## Datasets

All datasets used in our paper are available for selection. However, we only include a pre-trained checkpoint for the `adult` data in `checkpoint\`.

- adult
- airlines
- beijing
- credit_g
- default
- diabetes
- electricity
- kc1
- news
- nmes
- phoneme
- shoppers

Note that for the largest datasets (airlines and diabetes), detection score evaluation can take a considerable amount of time.


## Distributional Regression Tree (DT) Encoder

To encode $\mathbf{x}_{\text{num}}$ into $\mathbf{z}$, we utilize a distributional regression tree (DT). We train a separate DT for each feature, which partitions the feature's support and estimates a separate Gaussian distribution in each partition. Unlike a Gaussian Mixture model (GMM), the DT directly optimizes for a hard clustering given by the boundaries of the partitions. This reduces the overlap between Gaussian components and provably lowers the transport cost bound. Below, we illustrate the Gaussian components found by the DT encoder (with max depth = 3) for two features in the adult dataset. The red vertical lines represent the component-wise means.

![DTEncoder](https://github.com/muellermarkus/tabcascade/blob/master/figures/dt_groups.png)


## Provably lower transport cost bound

In the high-resolution model, our data-dependent coupling based on the DT encoder leads to a provably lower transport cost bound. As illustrated below, the source distribution is already very close to the data distribution.

![TransportCosts](https://github.com/muellermarkus/tabcascade/blob/master/figures/prob_paths.png)


## Citation

```
@inproceedings{mueller2026tabcascade,
  title = {Cascaded {{Flow Matching}} for {{Heterogeneous Tabular Data}} with {{Mixed-Type Features}}},
  booktitle = {International {{Conference}} on {{Machine Learning}}},
  author = {Mueller, Markus and Gruber, Kathrin and Fok, Dennis},
  year = {2026},
  url = {https://arxiv.org/abs/2601.22816}
}
```
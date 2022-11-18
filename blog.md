# Self-supervised Tensor Decomposition for Better Downstream Classification

This blog is written to introduce our recent NeurIPS 2022 paper: **ATD: Augmenting CP Tensor Decomposition by Self Supervision.** 

The paper introduces a new canonical polyadic tensor decomposition (CPD) approach empowered by self-supervised learning (SSL), which generates unsupervised embeddings that gives better downstream classification performance.

## A 5-min Summary 
Tensor decomposition can be used as a dimension reduction tool for downstream classification. However, traditional tensor decomposition methods focus on low-rank fitness and does not consider the downstream tasks (such as classification).

**Contribution 1:** This paper solves the problem of "how to learn tensor decomposition subspaces to generate better low-rank features for classification". We consider injecting class-preserving perturbations by tensor augmentation and then decomposing the tensor and the perturbed tensor together with the self-supervised loss!

<img src="framework.png"
     alt="ATD Framework"
     style="float: left; margin-right: 10px; " 
     width="800"/>

**Contribution 2:** For optimization, we improve the ALS algorithm for our new loss functions (including the non-convex self-supervised loss). Specifically, we build a new optimization algorithm that only uses least squares optimization and fix-point iteration for solving the non-convex subproblem.

Our method gives good results (with much fewer parameters) on human signal data, compared to contrastive learning methods, autoencoders, and other tensor decomposition methods.

---

We explain more details below.

## 1. Feature Dimension Reduction

**Unsupervised Learning:** Tensor decomposition and self-supervised learning are two types of unsupervised learning methods. They learn the encoders from unlabeled datasets and generate feature representations (e.g., 128-dim vectors) for each data sample. During the whole learning process, no label information is needed. 

**Downstream Evaluation:** The learned representations can be used as the input features for downstream classification with a separate linear model (e.g., logistic regression, which takes 128-dim vectors as input and predicts the label).

### 1.1 CP tensor decomposition

Canonical polyadic tensor decomposition (CPD) is commonly used to learn the low-rank factors of a tensor (e.g., higher-dimensional matrix). Standard CPD follows the **fitness principle**: approximating the original tensor as far as possible with the low-rank factors.

> Formally, assume the tensor is $\mathcal{T}\in\mathbb{R}^{I_1\times I_2\times\cdots\times I_K}$, the resulting $R$-rank factors can have the following dimensions: $\{A_i\in\mathbb{R}^{I_i\times R}: i\in[1,\dots,K]\}$. Frobenius norm is commonly chosen as the fitness loss for learning the low-rank factors.
> 

**Example:** we use multi-channel EEG signals for example. Assume each EEG signal has two channels (two blue time series in one slice). Now, we stack $N$ data samples (denoted as $T_1,T_2,...$) together and make them a 3-dimensional tensor $\mathcal{T}$: $N samples\times channels \times timesteps$.
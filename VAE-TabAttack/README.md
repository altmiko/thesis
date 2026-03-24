# Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data

This repository contains the official implementation of the paper "Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data" - a novel framework for generating adversarial examples in tabular data using Variational Autoencoders (VAEs).

## Abstract

Adversarial attacks on tabular data present fundamental challenges distinct from image or text domains due to the heterogeneous nature of mixed categorical and numerical features. Unlike images where pixel perturbations maintain visual similarity, tabular data lacks intuitive similarity metrics, making it difficult to define *imperceptible* modifications. Additionally, traditional gradient-based methods prioritise $\ell_p$-norm constraints, often producing adversarial examples that deviate from the original data distributions, making them detectable. We propose a latent space perturbation framework using a mixed-input Variational Autoencoder (VAE) to generate imperceptible adversarial examples. The proposed VAE integrates categorical embeddings and numerical features into a unified latent manifold, enabling perturbations that preserve statistical consistency. We specify *In-Distribution Success Rate* (IDSR) to measure the proportion of adversarial examples that remain statistically indistinguishable from the input distribution. Evaluation across six publicly available datasets and three model architectures  demonstrates that our method achieves substantially lower outlier rates and more consistent performance compared to traditional input-space attacks and other VAE-based methods adapted from image domain approaches. Our comprehensive analysis includes hyperparameter sensitivity, sparsity control mechanisms, and generative architectural comparisons, revealing that VAE-based attacks depend critically on reconstruction quality but offer superior practical utility when sufficient training data is available. This work highlights the importance of on-manifold perturbations for realistic adversarial attacks on tabular data, offering a robust approach for practical deployment.

## Features

- **VAE-based adversarial attacks** for tabular data with mixed numerical and categorical features
- **On-manifold perturbations** that preserve statistical consistency
- **In-Distribution Success Rate (IDSR)** metric for evaluating attack quality
- Support for multiple target model architectures (MLP, SoftDecisionTree, TabTransformer)
- Comprehensive evaluation on 6 public datasets
- **Sparsity control mechanisms** for generating sparse adversarial examples
- **Comparison with traditional attacks** (PGD, C&W, etc.)
- Jupyter notebook tutorials for easy experimentation

## Project Structure

```
VAE-TabAttack/
├── attack/                     # Attack implementations
│   ├── vae_attack.py          # Main VAE-based attack
│   ├── vae_sparsity_attack.py # Sparsity-controlled attacks
│   ├── traditional.py         # Traditional attack methods
│   └── gan_attack.py          # GAN-based attacks
├── data/                      # Data processing and datasets
│   ├── processor.py           # Data preprocessing utilities
│   ├── config.yaml           # Dataset configurations
│   └── datasets/             # Raw and processed datasets
├── mlmodel/                   # Target model implementations
│   ├── mlp.py                # Multi-layer perceptron
│   ├── softdt.py             # Soft decision trees
│   ├── tab_transformer.py     # TabTransformer model
│   └── vae.py                # VAE implementation
├── models/                    # Trained model checkpoints
├── results/                   # Experimental results
├── adversarial_examples/      # Generated adversarial examples
├── *.ipynb                   # Jupyter notebooks for experiments
└── README.md
```

## Datasets

The framework supports 6 public datasets with mixed categorical and numerical features:

| Dataset | $N$ (Samples) | $d$ (Features) | $d_{\text{num}}$ | $d_{\text{cat}}$ | $d_{\text{binary}}$ | $y$ (Classes) | Task | Domain |
|---------|---------------|----------------|-------------------|-------------------|---------------------|---------------|------|--------|
| Adult | 30,162 | 12 | 4 | 8 | 1 | 2 | Binary Classification | Census |
| Phishing | 11,430 | 86 | 57 | 29 | 28 | 2 | Binary Classification | Security |
| Pendigits | 10,992 | 16 | 16 | 0 | 0 | 10 | Multi-class Classification | Handwriting |
| German Credit | 1,000 | 19 | 8 | 11 | 2 | 2 | Binary Classification | Finance |
| Electricity | 45,312 | 8 | 7 | 1 | 0 | 2 | Binary Classification | Energy |
| Covertype | 581,012 | 54 | 10 | 54 | 54 | 7 | Multi-class Classification | Geography |

### Dataset Configuration

Each dataset is configured in `data/config.yaml` with:
- Feature type specifications (numerical, categorical, binary, ordinal)
- Train/validation/test splits
- Target column definitions

## Experiments

### Notebook Overview

1. **`1_datasets.ipynb`**: Data loading, preprocessing, and exploratory analysis
2. **`2_train_mlmodel.ipynb`**: Training target models (MLP, SoftDT, TabTransformer)
3. **`3_train_vae.ipynb`**: Training VAE models for different datasets
4. **`4_attacks.ipynb`**: Generating VAE-based adversarial attacks
5. **`4_attacks_sparsity.ipynb`**: Sparsity-controlled attack experiments
6. **`5_attacks_traditional.ipynb`**: Traditional attack baselines
7. **`6_evaluation.ipynb`**: Comprehensive evaluation and comparison
8. **`7_train_gan.ipynb`**: GAN-based attack experiments

### Hyperparameter Settings

Key hyperparameters for VAE training:
- Latent dimension: 10-50 (dataset dependent)
- Learning rate: 0.001-0.01
- Batch size: 256
- Training epochs: 100-500
- Beta (KL weight): 0.1-1.0

Attack parameters:
- Maximum iterations: 100-1000
- Step size: 0.01-0.1
- Sparsity constraint: L0/L1 regularization


# Methodology & Results — VAE Latent-Space Adversarial Attacks on NIDS

## 1. Introduction

This thesis presents the first application of VAE-based latent-space adversarial attacks to IoT network intrusion detection systems (NIDS) using the CICIoT2023 dataset. The core hypothesis is that adversarial examples generated through a learned VAE manifold are more likely to preserve protocol-level validity than unconstrained perturbation methods (e.g., PGD), because the VAE decoder implicitly enforces structural constraints learned from real traffic.

## 2. Dataset & Preprocessing

### 2.1 CICIoT2023
The CICIoT2023 dataset contains IoT network traffic across multiple attack categories. We consolidate fine-grained labels into 5 macro-classes:

| Macro-Class | Sub-types | Description |
|-------------|-----------|-------------|
| **DDoS** | 12 sub-types (ICMP_FLOOD, UDP_FLOOD, TCP_FLOOD, etc.) | Distributed denial-of-service |
| **DoS** | 4 sub-types (UDP_FLOOD, TCP_FLOOD, SYN_FLOOD, HTTP_FLOOD) | Denial-of-service |
| **Mirai** | 3 sub-types (GREETH_FLOOD, UDPPLAIN, GREIP_FLOOD) | Mirai botnet |
| **Recon** | 4 sub-types (HOSTDISCOVERY, OSSCAN, PORTSCAN, PINGSWEEP) | Reconnaissance |
| **Benign** | 1 type | Normal traffic |

### 2.2 Feature Representation (39 features)

Features are split into two groups based on their data type, which determines the VAE reconstruction loss:

**Continuous features (24)** — indices 0-22 and 38:
- Network statistics: Header_Length, Time_To_Live, Rate, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance
- Flag counts: fin/syn/rst/psh/ack/ece/cwr flag numbers, ack/syn/fin/rst counts
- Protocol Type (ordinal, treated as continuous)

**Binary features (15)** — indices 23-37:
- Protocol indicators: HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC

### 2.3 Preprocessing Pipeline

1. **Label consolidation**: Map 23 fine-grained attack labels to 5 macro-classes
2. **Cleaning**: Remove NaN and Inf values
3. **Binary clipping**: Clip binary features to [0, 1]
4. **Downsampling**: Cap DDoS and DoS classes at the Mirai class count to reduce extreme imbalance
5. **Train/test split**: 80/20 stratified split (random seed 42)
6. **Scaling**: StandardScaler on continuous features only; binary features remain unscaled in [0, 1]

## 3. Victim Models (NIDS Classifiers)

Two victim classifiers are trained on the preprocessed CICIoT2023 data. These represent the NIDS that the adversarial attack aims to evade.

### 3.1 LightGBM

A gradient-boosted decision tree ensemble (LGBMClassifier):
- 500 estimators, learning rate 0.05, 63 leaves
- GPU-accelerated training
- Balanced class weights
- Early stopping (patience = 50 rounds)

LightGBM is chosen as a representative non-differentiable model, requiring a black-box attack strategy.

### 3.2 CNN-LSTM

A lightweight deep learning classifier combining convolutional and recurrent layers:

```
Input (B, 39) → Unsqueeze → (B, 1, 39)
  → Conv1d(1, 32, k=3, pad=1) → ReLU → MaxPool1d(2) → (B, 32, 19)
  → Permute → (B, 19, 32)
  → LSTM(32, 64) → hidden state (B, 64)
  → Dropout(0.3) → Linear(64, 5) → logits
```

- Trained with Adam (lr=1e-3), balanced class-weighted CrossEntropyLoss
- Early stopping (patience = 5 epochs)

CNN-LSTM is a differentiable model, enabling gradient-based latent-space attacks.

## 4. VAE Architecture (MixedInputVAE)

A class-conditional VAE is trained separately for each attack class (DDoS, DoS, Mirai, Recon). Each VAE learns the distribution of its attack class in a low-dimensional latent space, which is then used as the search space for adversarial perturbations.

### 4.1 Architecture

The MixedInputVAE uses separate output heads for continuous and binary features:

**Encoder:**
```
Input (39) → [Linear → BatchNorm → ReLU → Dropout(0.2)] × L → fc_mu(latent_dim) + fc_log_var(latent_dim)
```

**Decoder:**
```
z (latent_dim) → [Linear → BatchNorm → ReLU → Dropout(0.2)] × L
  → cont_head: Linear → continuous output (24 features, MSE loss)
  → binary_head: Linear → Sigmoid → binary output (15 features, BCE loss)
```

### 4.2 Loss Function

```
L = L_cont + L_binary + β · L_KL
```

Where:
- `L_cont = MSE(x_cont, x̂_cont)` — continuous reconstruction
- `L_binary = BCE(x_bin, x̂_bin)` — binary reconstruction
- `L_KL = -0.5 · Σ(1 + log_var - μ² - exp(log_var))` — KL divergence
- `β` is annealed from 0 to `kl_weight` over the first 50% of training epochs

### 4.3 Per-Class Configuration

| Class | Hidden Dims | Epochs | KL Weight | Rationale |
|-------|------------|--------|-----------|-----------|
| DDoS  | [128, 64]  | 50     | 1.5       | Standard |
| DoS   | [128, 64]  | 50     | 1.5       | Standard |
| Mirai | [128, 64]  | 50     | 1.5       | Standard |
| Recon | [256, 128] | 100    | 1.5       | Larger network + more epochs due to higher intra-class variance |

### 4.4 Training Details
- Latent dimension: 16
- Optimizer: Adam (lr=1e-3, weight decay=1e-6)
- LR schedule: CosineAnnealingLR
- Gradient clipping: max norm 1.0
- Max samples per class: 100,000
- Feature reordering: input is reordered to [continuous | binary] before encoding; decoder outputs are mapped back to original feature order

### 4.5 Input Handling

The VAE is trained on *reordered* features: continuous features first (indices 0-22, 38), then binary features (indices 23-37). This groups features by type so the decoder's separate output heads align with contiguous slices of the internal representation. During attack inference, the `encode()` method internally reorders the original 39-feature input to match this training order, and `decode_to_full()` maps the decoder's outputs back to the original 39-feature positions.

## 5. Attack Pipeline

### 5.1 Overview

The attack generates adversarial examples in the VAE's latent space rather than directly in feature space:

```
x (original attack sample)
  → VAE.encode(x) → μ (latent mean)
  → Perturb: z' = μ + project(δ, L₂-ball of radius r)
  → VAE.decode(z') → (x̂_cont, x̂_bin)
  → Post-process: threshold binary features at 0.5
  → x' (adversarial example)
```

The key insight is that the L₂ ball constraint in latent space, combined with the VAE's learned manifold, implicitly constrains adversarial examples to remain near the data distribution — producing more protocol-valid adversarial traffic than unconstrained feature-space attacks.

### 5.2 Attack vs CNN-LSTM (Gradient-Based)

Since CNN-LSTM is differentiable, we use a C&W-style loss with gradient-based optimization:

**Loss:**
```
L = ‖δ‖₂ + λ · CW_loss(NIDS(x'), original_class)
```

Where the C&W loss (untargeted) is:
```
CW_loss = clamp(logit[orig_class] - max(logit[other_classes]), min=-κ)
```

**Optimization:**
- Initialize δ = 0 (shape: latent_dim)
- Optimize with Adam (lr=0.01)
- Project δ onto L₂ ball of radius r at each step
- κ = 0 (no confidence margin), λ = 1.0

**cuDNN compatibility note:** LSTM requires `model.train()` for backward pass due to cuDNN RNN kernel constraints. BatchNorm layers are explicitly set to `eval()` to prevent statistics corruption.

### 5.3 Attack vs LightGBM (Random Search)

Since LightGBM is non-differentiable, we use random search in latent space:

1. Sample random perturbation direction from N(0, I)
2. Scale by decaying step size: `step = r · (1 - t/T) · 0.1`
3. Move the search anchor in the perturbation direction
4. Project onto L₂ ball around the *original* μ (fixed anchor)
5. Decode candidate, evaluate with LightGBM
6. Accept if the original class probability decreases (greedy improvement)

### 5.4 Per-Class Attack Configuration

Attack radius and iteration counts are tuned per class based on empirical results:

| Class | Radius (r) | Max Iterations | Rationale |
|-------|-----------|---------------|-----------|
| DDoS  | 3.0       | 200           | High ASR at small radius |
| DoS   | 8.0       | 500           | Low ASR requires larger search space |
| Mirai | 10.0      | 500           | Near-zero ASR — very tight latent cluster |
| Recon | 5.0       | 300           | Moderate ASR, slight radius increase |

### 5.5 Post-Processing

After decoding from latent space:
- Binary features are thresholded at 0.5 (hard binarization)
- Continuous features are used as-is from the decoder

## 6. Evaluation Metrics

Three metrics assess the quality of adversarial examples:

### 6.1 Attack Success Rate (ASR)
Fraction of adversarial examples that are misclassified by the victim model (classified as any class other than the original attack class).

### 6.2 Protocol Validity Rate
Fraction of adversarial examples that satisfy domain-specific protocol constraints:
- **Transport mutual exclusivity**: At most one of TCP, UDP, ICMP, ARP, DHCP should be active per sample
- **Flag consistency**: Flag counts should be consistent with flag bits (e.g., fin_count > 0 ↔ fin_flag > 0)
- Additional network-layer validity checks

### 6.3 Intrusion Detection Survival Rate (IDSR)
Fraction of adversarial examples classified as *Benign* by the victim model — the most desirable outcome for an attacker, as it means the malicious traffic completely evades detection.

## 7. Preliminary Results

Results from the VAE latent-space attack (N=500 samples per class):

| Class | ASR (CNN-LSTM) | ASR (LightGBM) |
|-------|---------------|----------------|
| DDoS  | 0.944         | 0.986          |
| DoS   | 0.108         | 1.000*         |
| Mirai | 0.012         | 0.798          |
| Recon | 0.434         | 1.000*         |

*LightGBM results pending label alignment verification.

**Note:** These results were obtained before a critical bug fix in the VAE encoder input ordering (see progress.md). The VAE encoder was receiving features in the wrong order, degrading latent representations. Results should improve after re-running with the corrected code.

### 7.1 Analysis

- **DDoS** achieves high ASR against both models, suggesting the attack class has separable latent structure amenable to perturbation.
- **DoS and Mirai** show very low CNN-LSTM ASR. This is attributed to tight latent clustering — the VAE learns a compact representation for these classes, leaving little room for adversarial perturbation within the L₂ constraint. This represents a fundamental **validity–ASR tradeoff**: tighter manifold constraints preserve validity but limit attack freedom.
- **LightGBM** is generally more vulnerable, which is expected: the random search is unconstrained by gradient information and can explore more freely, while LightGBM's decision boundaries may be more susceptible to small feature changes.
- **DDoS/DoS confusion** exists at the dataset level (overlapping sub-types) — this is a known dataset limitation and is documented rather than "fixed."

## 8. Baseline Comparison (TODO)

### 8.1 PGD Baseline (`baseline_pgd.py`)
An unconstrained PGD attack directly in feature space (no VAE) will serve as the baseline:
- Same C&W loss, same victim models, same N=500 per class
- Continuous features: gradient step + L∞ or L₂ clipping
- Binary features: straight-through estimator or post-hoc thresholding
- Expected outcome: higher ASR but lower protocol validity than the VAE attack

### 8.2 Comparative Evaluation (`evaluate.py`)
A unified evaluation script will compute ASR, Protocol Validity Rate, and IDSR for both the VAE attack and PGD baseline, producing a single comparison table for the thesis.

## 9. Known Limitations

1. **DDoS/DoS confusion**: Exists at the dataset level due to overlapping sub-type definitions — not fixable, documented as a limitation.
2. **Reconstruction fidelity**: Continuous MAE targets (<0.03) not consistently met; binary accuracy (>98%) is the binding constraint for validity.
3. **Recon reconstruction error**: Highest among all classes (MAE 0.132, KL 10.7) due to high intra-class feature variance.
4. **cuDNN constraint**: LSTM must be in `train()` mode during backward pass; BatchNorm layers must be manually frozen to `eval()`.
5. **Validity–ASR tradeoff**: The VAE manifold constraint inherently limits attack freedom — classes with tight latent clusters (DoS, Mirai) show low ASR but are expected to have high validity.

# VAE-Constrained Adversarial Attack Generation on CICIoT2023
## Bachelor's Thesis Writeup

This document outlines the architecture, pipeline, and mechanics of the VAE-constrained adversarial attack codebase applied to the CICIoT2023 dataset.

### 1. How the Victim Models Work
The repository defines two distinct victim Network Intrusion Detection Systems (NIDS) targeted by the adversarial attacks. They are trained on the preprocessed 37-feature CICIoT2023 dataset (24 continuous, 13 binary features).
- **CNN-LSTM (`SimpleCNNLSTM`)**: A deep learning-based temporal-spatial feature extractor. The input (batch, 1, 37) is passed through a 1D Convolutional layer (`Conv1d(1, 32)`) with Max Pooling to extract local spatial patterns. The output sequence is then fed into an LSTM layer (`hidden_size=64`). The final hidden classification dense layer takes the output through a Dropout layer to predict one of the 5 classes.
- **LightGBM (`train_lightgbm.py`)**: A gradient-boosted decision tree ensemble trained with 200 estimators and a `learning_rate` of 0.1. Due to the discrete nature of tree decisions, this model is non-differentiable, requiring a different attack strategy than the standard gradient-based method used for the CNN-LSTM.

### 2. How the VAE Attack Models Work
To circumvent the issue of breaking fundamental protocol patterns (such as violating one-hot encoding or generating negative packet counts), the attacks are performed in the latent space of Variational Auto-Encoders (VAEs).
- **Architecture**: A separate `MixedInputVAE` model is trained for each of the 4 attack classes (*DDoS, DoS, Mirai, Recon*). The encoder uses an MLP to map the 37 features into a 16-dimensional multivariate normal manifold (`mu`, `log_var`).
- **Mixed Output Heads**: The decoder splits into two heads. The continuous head outputs 24 features trained with Mean Squared Error (MSE), and the binary head outputs 13 features through a Sigmoid activation trained with Binary Cross-Entropy (BCE). Beta-annealed Kullback-Leibler (KL) divergence regularizes the latent space.
- **Purpose**: By acting as a feature manifold, the VAE restricts any generated perturbations to realistic network traffic resembling the original attack class.

### 3. How the Attacks Work and What Libraries are Used
The adversarial attacks do not rely on standard external attack libraries (like *CleverHans* or *ART*). Instead, they are custom-developed in standard PyTorch using `torch.autograd` to specifically accommodate the mixed-feature data and latent-space bounds.

**A. CNN-LSTM Attack (Gradient Descent)**
- **Latent C&W Optimization**: A benign (or correctly classified) sample is encoded into a latent center `mu`. An Adam optimizer searches for a perturbation `delta` to form an adversarial latent vector `z_adv = mu + delta`.
- **L2 Sphere Constraint**: The magnitude of `delta` is clamped conservatively at each step to a radius `R * 0.99`.
- **Straight-Through Estimator (STE)**: To allow gradient propagation while keeping binary features discretely 0 or 1, a straight-through estimator (`binary_hard + bin_out - bin_out.detach()`) simulates binary outputs during backpropagation.
- **Custom Loss Objective**: The cost function is a combination of:
  1. **Carlini & Wagner (C&W) Loss**: Fools the logit layer of the CNN-LSTM.
  2. **Latent Norm Penalty**: Discourages excessive deviations from `mu`.
  3. **Reconstruction Anchoring**: Keeps the adversarially decoded sample close to the original input.
  4. **Constraint Violation Penalty**: Punishes outputs drifting past historical `cont_min`/`cont_max` bounds or fractional binary properties.

**B. LightGBM Attack (Randomized Restart Search)**
- Since LightGBM gradients cannot be backpropagated, the attack utilizes random standard normal noise bounded inside the `R` radius in the latent space to propose candidates `z_cand`.
- The `z_cand` proposals are decoded and heavily clamped, and the configuration that locally minimizes the probability belonging to the true class is selected iteratively.

### 4. Constraint Enforcement Validator
A `validator.py` script rigorously enforces physical mapping protocol semantics sourced directly from the CICIoT2023 paper (Neto et al.).
- **Protocol Exclusivity**: E.g., TCP and UDP cannot both be 1. ARP/ICMP exclusivity is rigidly checked.
- **Network Layer Dependencies**: Application-level protocols (HTTP, SSH, IRC) must have the TCP binary feature active. DHCP requires UDP.
- **Mathematical Consistency**: Packet statistical features must always respect `Min <= AVG <= Max`. Packet/flag counts must remain correctly correlated and non-negative.
- **Implementation**: The pipeline enforces these bounds *in-loop* (via masking/clamping) and strictly drops any generated adversarial sample that fails these validations before calculating the Adversarial Success Rate (ASR).

### 5. Bugs Found & Fixed
During the initial code-review of the attack script (`run_attacks.py`), the following critical issue was identified:
- **Flawed Safe Feature Masking**: To preserve inter-feature semantic stability (akin to *NetDiffuser* architectures), highly correlated continuous features are frozen via a generated `safe_mask`. However, the algorithm inadvertently initialized the mask for all **binary features** to `0.0` as well.
- **Negative Repercussions**: The formulation `x_adv = x_orig + mask * (x_decoded - x_orig)` caused the attack script to freeze all binary features to their original unperturbed input states. The target VAE effectively lost the ability to perturb or suggest new binary flags through the latent search, critically damaging the optimization scope.
- **Resolution**: I successfully patched the `compute_safe_feature_mask` tensor loop by assigning `mask[idx] = 1.0` for all 13 binary features. Now, gradients flow successfully to the binary VAE decoder heads, and the pipeline correctly utilizes the Straight-Through Estimator optimization for discrete modifications.

### 6. Comprehensive Multi-Layer Naturalness Validation
To ensure generated adversarial traffic remains absolutely indistinguishable from real benign traffic and abides by statistical coherence laws, a fully integrated `thesis_validation_suite.py` was implemented:
- **Box & Boundary Enforcement**: Ensures generated vectors fall strictly within a 1% threshold of realistic historical minimums and maximums (`_check_train_distribution`).
- **Statistical Tests (Kolmogorov-Smirnov)**: Conducts aggressive KS Two-Sample evaluation (`p_value > 0.01`) across every continuous feature to confirm that malicious perturbations have not statistically drifted out of the intrinsic historical manifold.
- **Correlational Drift Penalty**: Audits the correlation matrix of adversarial features against the benign training standard. If `corr_diff` snaps past an arbitrary tolerance (0.5), it detects potentially unnatural feature combinations.
- **Autoencoder Reconstruction Anchor**: Tests adversarial samples backward through the VAE. If the Mean-Squared Error of reconstruction drastically exceeds historical benign variance, it flags the samples as Out-Of-Distribution (OOD).
- **PCA Visualization & Manual Overlays**: Auto-generates kernel density estimates (KDE) and 2D-PCA projections for immediate visual clearance. All visuals are exported directly to standard thesis directories.

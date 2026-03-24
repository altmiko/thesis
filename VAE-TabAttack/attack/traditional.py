import torch
import torch.nn as nn
import torch.optim as optim

def random_noise_attack(data, epsilon, noise_type="uniform", one_hot_indices=None, binary_indices=None):
    """
    Perform a random noise attack with handling for one-hot encoded and binary features.

    Parameters:
        data: Input data tensor.
        epsilon: Maximum magnitude of the noise.
        noise_type: Type of noise to add ("uniform" or "normal").
        one_hot_indices: List of tuples (start, end) representing ranges of one-hot encoded features.
        binary_indices: List of column indices representing binary features.

    Returns:
        perturbed_data: Adversarially perturbed data with random noise.
    """
    # Generate random noise
    if noise_type == "uniform":
        noise = torch.empty_like(data).uniform_(-epsilon, epsilon)
    elif noise_type == "normal":
        noise = torch.empty_like(data).normal_(mean=0, std=epsilon / 2)  # Adjust std for desired range
    else:
        raise ValueError("Unsupported noise_type. Choose 'uniform' or 'normal'.")

    # Add noise to the input data
    perturbed_data = data + noise

    # Clip the perturbed data to ensure it's within [0,1] range
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    # Adjust for one-hot encoded features
    if one_hot_indices:
        perturbed_data = get_one_hot_indices(perturbed_data, one_hot_indices)

    # Adjust for binary features
    if binary_indices:
        perturbed_data = adjust_binary_features(perturbed_data, binary_indices)

    return perturbed_data



def fgsm_attack(model, data, target, epsilon, one_hot_indices=None, binary_indices=None):
    """
    Perform FGSM attack (non-targeted) with handling for one-hot encoded and binary features.

    Parameters:
        model: PyTorch model to attack.
        data: Input data tensor.
        target: Target labels tensor.
        epsilon: Perturbation magnitude.
        one_hot_indices: List of tuples (start, end) representing ranges of one-hot encoded features.
        binary_indices: List of column indices representing binary features.

    Returns:
        perturbed_data: Adversarially perturbed data with valid one-hot and binary features.
    """
    # Ensure input tensor requires gradients
    data.requires_grad = True

    # Forward pass through the model
    output = model(data)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Backward pass to calculate gradients of loss w.r.t. input data
    loss.backward()

    # Collect the sign of the gradients
    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()

    # Create perturbed data by adding the sign of the gradient times epsilon
    perturbed_data = data + epsilon * sign_data_grad

    # Clip the perturbed data to ensure it's within [0,1] range
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    # Adjust for one-hot encoded features
    if one_hot_indices:
        perturbed_data = get_one_hot_indices(perturbed_data, one_hot_indices)

    # Adjust for binary features
    if binary_indices:
        perturbed_data = adjust_binary_features(perturbed_data, binary_indices)

    return perturbed_data



def pgd_attack(model, data, target, epsilon, alpha, num_iter, one_hot_indices=None, binary_indices=None):
    """
    Perform PGD attack (non-targeted) with handling for one-hot encoded and binary features.

    Parameters:
        model: PyTorch model to attack.
        data: Input data tensor.
        target: Target labels tensor.
        epsilon: Perturbation magnitude.
        alpha: Step size for each iteration.
        num_iter: Number of iterations to perform.
        one_hot_indices: List of tuples (start, end) representing ranges of one-hot encoded features.
        binary_indices: List of column indices representing binary features.

    Returns:
        perturbed_data: Adversarially perturbed data with valid one-hot and binary features.
    """
    # Make a copy of the data to perturb
    perturbed_data = data.detach().clone()
    perturbed_data.requires_grad = True

    for _ in range(num_iter):
        # Forward pass through the model
        output = model(perturbed_data)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Backward pass to calculate gradients of loss w.r.t. input data
        loss.backward()

        # Collect the gradients
        data_grad = perturbed_data.grad.data

        # Perform the PGD step
        with torch.no_grad():
            perturbed_data = perturbed_data + alpha * data_grad.sign()

            # Clip the perturbed data to ensure it's within the epsilon-ball of the original data
            perturbed_data = torch.clamp(perturbed_data, data - epsilon, data + epsilon)

            # Clip to ensure the perturbed data is within [0,1]
            perturbed_data = torch.clamp(perturbed_data, 0, 1)

        # Detach and reset gradients for the next iteration
        perturbed_data = perturbed_data.detach().clone()
        perturbed_data.requires_grad = True

    # Adjust for one-hot encoded features
    if one_hot_indices:
        perturbed_data = get_one_hot_indices(perturbed_data, one_hot_indices)
    
    # Adjust for binary features
    if binary_indices:
        perturbed_data = adjust_binary_features(perturbed_data, binary_indices)

    return perturbed_data



def get_one_hot_indices(perturbed_data, one_hot_indices):
    """
    Adjust one-hot encoded features to ensure a valid one-hot encoding.

    Parameters:
        perturbed_data: Tensor containing perturbed data.
        one_hot_indices: List of tuples (start, end) representing the ranges of one-hot encoded groups.

    Returns:
        adjusted_data: Tensor with valid one-hot encoded features.
    """
    # Clone the perturbed data to avoid modifying the original
    adjusted_data = perturbed_data.clone()

    for start, end in one_hot_indices:
        # Extract the slice corresponding to one-hot encoded features
        one_hot_slice = adjusted_data[:, start:end]

        # Identify the index of the maximum value for each data point in the slice
        max_indices = torch.argmax(one_hot_slice, dim=1)

        # Zero out all elements in the one-hot slice
        one_hot_slice.zero_()

        # Set the maximum index to 1 (reconstruct valid one-hot encoding)
        one_hot_slice[torch.arange(one_hot_slice.size(0)), max_indices] = 1.0

        # Assign the modified slice back to the adjusted data
        adjusted_data[:, start:end] = one_hot_slice

    return adjusted_data


def adjust_binary_features(perturbed_data, binary_indices):
    # Clone the perturbed data to avoid modifying the original
    adjusted_data = perturbed_data.clone()

    start, end = binary_indices
    for idx in range(start, end, 1):
        # Round the feature value to the nearest valid binary value (0 or 1)
        adjusted_data[:, idx] = (adjusted_data[:, idx] >= 0.5).float()

    return adjusted_data


def attack_success_rate(model, original_data, perturbed_data, original_labels):
    """
    Calculate the attack success rate.

    Parameters:
        model: PyTorch model to evaluate.
        original_data: Original input data tensor.
        perturbed_data: Adversarially perturbed data tensor.
        original_labels: True labels for the original data.

    Returns:
        success_rate: Proportion of successful attacks.
    """
    # Get predictions on original and perturbed data
    original_preds = model(original_data).argmax(dim=1)
    perturbed_preds = model(perturbed_data).argmax(dim=1)

    # Successful attacks occur when the perturbed prediction is different from the original label
    successful_attacks = (perturbed_preds != original_labels).sum().item()
    success_rate = successful_attacks / original_labels.size(0)

    return success_rate

# Example usage
if __name__ == "__main__":
    # Assume a simple model, data, and target tensors for demonstration purposes
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(28 * 28, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)  # Flatten the input
            return self.fc(x)

    model = SimpleModel()
    loss_fn = nn.CrossEntropyLoss()

    # Example data and target
    data = torch.rand((1, 1, 28, 28))  # Random 28x28 input
    target = torch.tensor([3])  # Example target label

    # Set epsilon, alpha, and number of iterations
    epsilon = 0.1
    alpha = 0.01
    num_iter = 10

    # Perform FGSM attack
    perturbed_data_fgsm = fgsm_attack(model, loss_fn, data, epsilon)
    print("FGSM Perturbed Data:", perturbed_data_fgsm)

    # Perform PGD attack
    perturbed_data_pgd = pgd_attack(model, loss_fn, data, epsilon, alpha, num_iter)
    print("PGD Perturbed Data:", perturbed_data_pgd)




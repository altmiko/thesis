import os
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

class VAEDeltaZAttack:
    def __init__(self, 
                 ml_model: nn.Module,
                 dataset: str, 
                 model: str,
                 p_norm: int = 1,        # Norm type for regularization (1=L1, 2=L2)
                 lambda_: float = 0.1,    # Weight for regularization term
                 lr: float = 0.01,  # Learning rate for deltaZ optimization
                 max_iter: int = 300,   # Number of iterations to optimize deltaZ
                 vae_model: nn.Module = None,
                 batch_size: int = 16,
                 device: str = "cpu"
                 ):
        self._mlmodel = ml_model        # Classification model
        self._dataset = dataset          # Dataset name
        self._model = model              # Model name
        
        self._p_norm = p_norm            # Norm for regularization (L1 or L2)
        self._lambda = lambda_           # Weight for regularization
        self._lr = lr         # Learning rate
        self._max_iter = max_iter  # Iterations to learn deltaZ

        self.vae = vae_model
        self.device = device
        self._batch_size = batch_size
        self._attack = "deltaz"
        
        # Initialize deltaZ (will be optimized later)
        self._delta_z = None

    def get_adversarial_examples(
        self, 
        factuals: torch.Tensor, 
        folder: str = "adversarial_examples"
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[np.ndarray], List[float]]:
        """
        Generate adversarial examples using deltaZ attack in the latent space.
        
        Args:
            factuals: Original input data.
            folder: Folder to save the adversarial examples.
            
        Returns:
            Tuple containing:
                - List of adversarial examples
                - List of latent vectors
                - List of booleans indicating attack success
                - List of original latent vectors
                - List of confidences for the adversarial examples
        """
        # Create directories for storing results
        sample_num = factuals.shape[0]
        folder = os.path.join(folder, 
                            os.path.join(os.path.join(f"{self._dataset}", f"{self._model}"), 
                            f"{self._attack}"))
        folder = os.path.join(folder, f"Try_{sample_num}_inputs")
        folder = os.path.join(folder, 
                            f"_lambda_{self._lambda}_p_{self._p_norm}_lr_{self._lr}_max_iter_{self._max_iter}")
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            print(f"Folder {folder} already exists. Overwriting...")

        # First, learn the delta_z if it doesn't exist
        if self._delta_z is None:
            self._learn_delta_z(factuals)
        
        # Generate adversarial examples using the learned delta_z
        adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences = self._delta_z_attack(factuals, folder)
        
        # Print attack statistics
        total_successful_attacks = sum(attack_success_list)
        print(f"Total successful adversarial examples: {total_successful_attacks}/{len(factuals)} ({total_successful_attacks/len(factuals)*100:.2f}%)")
        print(f"Adversarial examples confidence: Mean - {np.mean(confidences):.4f}, Max - {np.max(confidences):.4f}, Min - {np.min(confidences):.4f}")

        return adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences

    def _learn_delta_z(self, factuals: torch.Tensor):
        """
        Learn a single deltaZ that can flip class predictions for inputs.
        
        Args:
            factuals: Original input data used to learn deltaZ.
        """
        print("Learning deltaZ transformation vector...")
        
        # Prepare data for optimization
        train_loader = torch.utils.data.DataLoader(
            factuals, batch_size=self._batch_size, shuffle=True
        )
        
        # Initialize deltaZ randomly
        latent_dim = self.vae._mu_enc.out_features  # Get latent dimension from VAE
        self._delta_z = torch.randn(1, latent_dim, device=self.device, requires_grad=True)
        
        # Create optimizer for deltaZ
        optimizer = torch.optim.Adam([self._delta_z], lr=self._lr)
        
        # Training loop for deltaZ
        with tqdm(total=self._max_iter, desc="Learning DeltaZ") as pbar:
            for iteration in range(self._max_iter):
                running_loss = 0.0
                num_batches = 0
                
                for batch_idx, query_instances in enumerate(train_loader):
                    if isinstance(query_instances, list) and len(query_instances) == 2:
                        # If data loader returns (x, y) pairs
                        query_instances, _ = query_instances
                        
                    query_instances = query_instances.float().to(self.device)
                    
                    # Get original predictions
                    original_outputs = self._mlmodel.forward(query_instances)
                    original_classes = torch.argmax(original_outputs, dim=1)
                    
                    # Encode inputs to get latent vectors
                    x_cat = query_instances[:, :self.vae.num_categorical].long()
                    x_num = query_instances[:, self.vae.num_categorical:]
                    mu, logvar = self.vae.encode(x_cat, x_num)
                    
                    # Sample from latent distribution
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                    
                    # Apply deltaZ (flipping direction based on class)
                    # For binary classification:
                    # If class 0, add deltaZ to move to class 1
                    # If class 1, subtract deltaZ to move to class 0
                    direction = (1 - 2 * original_classes).float().view(-1, 1)
                    z_perturbed = z + direction * self._delta_z
                    
                    # Decode perturbed latent vectors
                    adv_examples, _, _ = self.vae.decode(z_perturbed)
                    
                    # Get predictions for perturbed examples
                    adv_outputs = self._mlmodel.forward(adv_examples)
                    
                    # Loss: want to flip predictions and keep deltaZ small
                    target_classes = 1 - original_classes  # Flip the targets
                    
                    # Classification loss (want to maximize probability of target class)
                    clf_loss = F.cross_entropy(adv_outputs, target_classes)
                    
                    # Regularization loss (L1 or L2 norm of deltaZ)
                    reg_loss = torch.norm(self._delta_z, p=self._p_norm)
                    
                    # Total loss
                    loss = clf_loss + self._lambda * reg_loss
                    
                    # Update deltaZ
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx >= 10:  # Use a limited number of batches per iteration
                        break
                
                # Update progress bar
                avg_loss = running_loss / max(1, num_batches)
                pbar.set_postfix(loss=f"{avg_loss:.6f}", reg=f"{reg_loss.item():.6f}")
                pbar.update(1)
        
        print(f"DeltaZ learned with norm: {torch.norm(self._delta_z, p=self._p_norm).item():.6f}")

    def _delta_z_attack(
        self, df_fact: torch.Tensor, folder: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[np.ndarray], List[float]]:
        """
        Apply the learned deltaZ to generate adversarial examples.
        
        Args:
            df_fact: Original input data.
            folder: Folder to save the results.
        
        Returns:
            Tuple of adversarial examples, latent vectors, attack success indicators, 
            original latent vectors, and confidences.
        """
        # Prepare data
        test_loader = torch.utils.data.DataLoader(
            df_fact, batch_size=self._batch_size, shuffle=False
        )

        adversarial_examples = []
        latent_vectors = []
        original_latent_vectors = []
        attack_success_list = []
        confidences = []

        with tqdm(total=len(df_fact), desc="Generating DeltaZ Adversarial Examples") as pbar:
            for batch_idx, query_instances in enumerate(test_loader):
                query_instances = query_instances.float().to(self.device)

                # Get the original class predictions
                original_outputs = self._mlmodel.forward(query_instances)
                original_classes = torch.argmax(original_outputs, dim=1).detach().cpu().numpy()

                # Split input into categorical and numerical parts
                x_cat = query_instances[:, :self.vae.num_categorical].long()
                x_num = query_instances[:, self.vae.num_categorical:]

                # Get initial latent representation
                z_init, logvar = self.vae.encode(x_cat, x_num)
                
                # Process each instance in the batch
                for i in range(query_instances.size(0)):
                    instance_idx = batch_idx * self._batch_size + i
                    
                    # Get the original instance and its latent vector
                    query_instance = query_instances[i].unsqueeze(0)
                    z_init_single = z_init[i].detach().clone()
                    original_class = original_classes[i]
                    
                    # Apply deltaZ based on original class
                    direction = 1 if original_class == 0 else -1
                    z_perturbed = z_init_single + direction * self._delta_z[0]
                    
                    # Decode perturbed latent vector
                    adv_example, _, _ = self.vae.decode(z_perturbed.unsqueeze(0))
                    
                    # Check if the attack was successful
                    output = self._mlmodel.forward(adv_example)
                    predicted_class = torch.argmax(output, dim=1).item()
                    is_success = (predicted_class != original_class)
                    
                    # Calculate confidence of not being in the original class
                    softmax_probs = F.softmax(output, dim=1)
                    confidence_not_original = 1 - softmax_probs[0, original_class].item()
                    
                    # Save results
                    adversarial_examples.append(adv_example.detach().cpu().numpy())
                    latent_vectors.append(z_perturbed.detach().cpu().numpy())
                    attack_success_list.append(is_success)
                    original_latent_vectors.append(z_init_single.detach().cpu().numpy())
                    confidences.append(confidence_not_original)
                    
                    # Save results to files
                    self._save_npy_files(
                        idx=instance_idx,
                        folder=folder,
                        adversarial_example=adv_example.detach().cpu().numpy(),
                        latent_vector=z_perturbed.detach().cpu().numpy(),
                        attack_success=is_success,
                        original_instance=query_instance.detach().cpu().numpy(),
                        original_latent_vector=z_init_single.detach().cpu().numpy(),
                        confidence=confidence_not_original
                    )
                    
                    pbar.update(1)
                    
        return adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences

    def visualize_delta_z(self, save_path=None):
        """
        Visualize the learned deltaZ vector.
        
        Args:
            save_path: Path to save the visualization. If None, just display.
        """
        import matplotlib.pyplot as plt
        
        if self._delta_z is None:
            print("DeltaZ has not been learned yet.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.stem(self._delta_z.detach().cpu().numpy()[0])
        plt.title("DeltaZ Vector")
        plt.xlabel("Latent Dimension")
        plt.ylabel("Value")
        
        if save_path:
            plt.savefig(save_path)
            print(f"DeltaZ visualization saved to {save_path}")
        else:
            plt.show()
    
    def evaluate_attack(self, original_data, adversarial_examples, original_classes=None):
        """
        Evaluate the success of the adversarial attack.
        
        Args:
            original_data: Original input data
            adversarial_examples: Generated adversarial examples
            original_classes: Original class labels (if None, will be computed)
            
        Returns:
            Dictionary with attack metrics
        """
        # Convert to tensors if needed
        if not isinstance(original_data, torch.Tensor):
            original_data = torch.tensor(original_data, dtype=torch.float).to(self.device)
        
        if not isinstance(adversarial_examples, torch.Tensor):
            adversarial_examples = torch.tensor(adversarial_examples, dtype=torch.float).to(self.device)
        
        # Get original predictions if not provided
        if original_classes is None:
            original_outputs = self._mlmodel.forward(original_data)
            original_classes = torch.argmax(original_outputs, dim=1)
        
        # Get predictions for adversarial examples
        adv_outputs = self._mlmodel.forward(adversarial_examples)
        adv_classes = torch.argmax(adv_outputs, dim=1)
        
        # Calculate success rate
        success_rate = (adv_classes != original_classes).float().mean().item()
        
        # Calculate L2 distance in input space
        l2_distances = torch.norm(adversarial_examples - original_data, p=2, dim=1)
        avg_l2_distance = l2_distances.mean().item()
        
        # Calculate L∞ distance in input space
        linf_distances = torch.norm(adversarial_examples - original_data, p=float('inf'), dim=1)
        avg_linf_distance = linf_distances.mean().item()
        
        # Return metrics
        return {
            'success_rate': success_rate * 100,  # as percentage
            'avg_l2_distance': avg_l2_distance,
            'avg_linf_distance': avg_linf_distance,
            'max_l2_distance': l2_distances.max().item(),
            'max_linf_distance': linf_distances.max().item(),
            'delta_z_norm': torch.norm(self._delta_z, p=self._p_norm).item()
        }

    def _save_npy_files(self, idx: int, folder: str, adversarial_example: np.ndarray, latent_vector: np.ndarray, 
                        attack_success: bool, original_instance: np.ndarray, original_latent_vector: np.ndarray,
                        confidence: float):
        """
        Save data for a single factual as .npy files.
        
        Args:
            idx: Index of the factual
            folder: Folder to save the files
            adversarial_example: Generated adversarial example
            latent_vector: Latent vector of the adversarial example
            attack_success: Whether the attack was successful
            original_instance: Original factual instance
            original_latent_vector: Original latent vector
            confidence: Confidence of the adversarial example
        """
        np.save(os.path.join(folder, f"adversarial_example_{idx}.npy"), adversarial_example)
        np.save(os.path.join(folder, f"latent_vector_{idx}.npy"), latent_vector)
        np.save(os.path.join(folder, f"attack_success_{idx}.npy"), np.array(attack_success, dtype=bool))
        np.save(os.path.join(folder, f"original_instance_{idx}.npy"), original_instance)
        np.save(os.path.join(folder, f"original_latent_vector_{idx}.npy"), original_latent_vector)
        np.save(os.path.join(folder, f"confidence_{idx}.npy"), np.array(confidence, dtype=float))


# Example usage
if __name__ == "__main__":
    # This is a placeholder for demonstration - you would adapt this to your actual usage
    
    import torch.nn as nn
    
    # Load your ML model
    ml_model = None  # Replace with your actual ML model
    
    # Load your VAE model
    vae_model = None  # Replace with your actual VAE model
    
    # Create the DeltaZ attack
    deltaz_attack = VAEDeltaZAttack(
        ml_model=ml_model,
        dataset="your_dataset",
        model="your_model",
        p_norm=1,              # L1 norm for regularization
        lambda_=0.1,           # Regularization weight
        lr=0.01,    # Learning rate for optimization
        max_iter=1000,   # Number of iterations to learn deltaZ
        vae_model=vae_model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load your data
    data = None  # Replace with your actual data
    
    # Generate adversarial examples
    adv_examples, latent_vectors, success_list, original_latents, confidences = deltaz_attack.get_adversarial_examples(data)
    
    # Visualize deltaZ
    deltaz_attack.visualize_delta_z("deltaz_visualization.png")
    
    # Evaluate the attack
    metrics = deltaz_attack.evaluate_attack(data, adv_examples)
    print(f"Attack metrics: {metrics}")
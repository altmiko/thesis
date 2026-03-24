import os
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

class VAEPGDAttack:
    def __init__(self, 
                 ml_model: nn.Module,
                 dataset: str, 
                 model: str,
                 epsilon: float = 0.1,  # Perturbation constraint
                 alpha: float = 0.01,   # Step size for PGD
                 num_steps: int = 10,   # Number of steps for PGD
                 vae_model: nn.Module = None,
                 batch_size: int = 16,
                 random_start: bool = True,  # Whether to start with random perturbation
                 device: str = "cpu"
                 ):
        self._mlmodel = ml_model  # Classification model
        self._dataset = dataset    # Dataset name
        self._model = model        # Model name

        self._epsilon = epsilon    # Perturbation bound
        self._alpha = alpha        # Step size for PGD
        self._num_steps = num_steps
        self._random_start = random_start

        self.vae = vae_model
        self.device = device
        self._batch_size = batch_size
        self._attack = "pgd"
        self._convergence_threshold = 1e-5

    def get_adversarial_examples(
        self, 
        factuals: torch.Tensor, 
        folder: str = "adversarial_examples"
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[np.ndarray], List[float]]:
        """
        Generate adversarial examples using PGD attack in the latent space.
        
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
                            f"_epsilon_{self._epsilon}_alpha_{self._alpha}_steps_{self._num_steps}")
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            print(f"Folder {folder} already exists. Overwriting...")

        # Generate adversarial examples
        adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences = self._pgd_attack(factuals, folder)
        
        # Print attack statistics
        total_successful_attacks = sum(attack_success_list)
        print(f"Total successful adversarial examples: {total_successful_attacks}/{len(factuals)} ({total_successful_attacks/len(factuals)*100:.2f}%)")
        print(f"Adversarial examples confidence: Mean - {np.mean(confidences):.4f}, Max - {np.max(confidences):.4f}, Min - {np.min(confidences):.4f}")

        return adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences

    # Removed optimizer method as PGD handles updates directly

    def _pgd_attack(
        self, df_fact: torch.Tensor, folder: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[np.ndarray], List[float]]:
        """
        Implement PGD attack in the latent space of the VAE.
        
        Args:
            df_fact: Original input data.
            folder: Folder to save the results.
        
        Returns:
            Tuple of adversarial examples, latent vectors, attack success indicators, 
            original latent vectors, and confidences.
        """
        # Prepare data for optimization steps
        test_loader = torch.utils.data.DataLoader(
            df_fact, batch_size=self._batch_size, shuffle=False
        )

        adversarial_examples = []
        latent_vectors = []
        original_latent_vectors = []
        attack_success_list = []
        confidences = []

        with tqdm(total=len(df_fact), desc="Generating PGD Adversarial Examples") as pbar:
            for batch_idx, query_instances in enumerate(test_loader):
                query_instances = query_instances.float().to(self.device)

                # Get the original class predictions
                original_outputs = self._mlmodel.forward(query_instances)
                original_classes = torch.argmax(original_outputs, dim=1).detach().cpu().numpy()

                # Split input into categorical and numerical parts
                x_cat = query_instances[:, :self.vae.num_categorical].long()
                x_num = query_instances[:, self.vae.num_categorical:]

                # Get initial latent representation
                z_init, _ = self.vae.encode(x_cat, x_num)
                
                # Process each instance in the batch
                for i in range(query_instances.size(0)):
                    instance_idx = batch_idx * self._batch_size + i
                    
                    # Get the original instance and its latent vector
                    query_instance = query_instances[i].unsqueeze(0)
                    z_init_single = z_init[i].detach().clone()
                    original_class = original_classes[i]
                    
                    # Initialize z with random perturbation if random_start is True
                    if self._random_start:
                        random_noise = torch.FloatTensor(z_init_single.shape).uniform_(-self._epsilon, self._epsilon).to(self.device)
                        z_single = torch.clamp(z_init_single + random_noise, min=z_init_single - self._epsilon, max=z_init_single + self._epsilon)
                    else:
                        z_single = z_init_single.clone()
                    
                    best_adv_example = None
                    best_latent_vector = None
                    best_confidence = 0.0
                    attack_success = False
                    

                    # Reset gradients for the optimization step
                    z_single = z_single.detach().requires_grad_(True)
                    # PGD attack loop              
                    # PGD steps
                    for step in range(self._num_steps):
                        # Decode the latent vector to get the adversarial example
                        adv_example, _, _ = self.vae.decode(z_single.unsqueeze(0))
                        
                        # Get model predictions for the adversarial example
                        output = self._mlmodel.forward(adv_example)
                        
                        # Compute the adversarial loss
                        loss = self._compute_adversarial_loss(z_single, z_init_single, output, original_class)
                        
                        # Compute gradients
                        loss.backward()
                        
                        # Update the latent vector using the gradient
                        with torch.no_grad():
                            # Take a step in the gradient direction
                            grad = z_single.grad.sign()  # Sign of the gradient for FGSM-like updates
                            z_single = z_single - self._alpha * grad  # Minimize loss
                            
                            # Project back to the ε-ball around the original latent vector
                            delta = z_single - z_init_single
                            delta = torch.clamp(delta, min=-self._epsilon, max=self._epsilon)
                            z_single = z_init_single + delta
                        
                        # Reset gradients for the next step
                        if step < self._num_steps - 1:
                            z_single = z_single.detach().requires_grad_(True)
                    
                    # Decode the final latent vector to get the adversarial example
                    adv_example, _, _ = self.vae.decode(z_single.unsqueeze(0))
                    output = self._mlmodel.forward(adv_example)
                    predicted_class = torch.argmax(output, dim=1).item()
                    
                    # Check if the attack was successful (misclassification)
                    is_success = (predicted_class != original_class)
                    
                    # Calculate confidence of not being in the original class
                    softmax_probs = F.softmax(output, dim=1)
                    confidence_not_original = 1 - softmax_probs[0, original_class].item()
                    
                    # Update the best adversarial example if current one is better
                    if is_success and (best_adv_example is None or confidence_not_original > best_confidence):
                        best_adv_example = adv_example.detach().cpu().numpy()
                        best_latent_vector = z_single.detach().cpu().numpy()
                        best_confidence = confidence_not_original
                        attack_success = True
                    elif not is_success and (best_adv_example is None or confidence_not_original > best_confidence):
                        best_adv_example = adv_example.detach().cpu().numpy()
                        best_latent_vector = z_single.detach().cpu().numpy()
                        best_confidence = confidence_not_original
                    
                    # If no successful attack was found, use the highest confidence example
                    if best_adv_example is None:
                        best_adv_example = query_instance.detach().cpu().numpy()
                        best_latent_vector = z_init_single.detach().cpu().numpy()
                        best_confidence = 0.0
                        attack_success = False
                    
                    # Append results for this instance
                    adversarial_examples.append(best_adv_example)
                    latent_vectors.append(best_latent_vector)
                    attack_success_list.append(attack_success)
                    original_latent_vectors.append(z_init_single.detach().cpu().numpy())
                    confidences.append(best_confidence)
                    
                    # Save results to files
                    self._save_npy_files(
                        idx=instance_idx,
                        folder=folder,
                        adversarial_example=best_adv_example,
                        latent_vector=best_latent_vector,
                        attack_success=attack_success,
                        original_instance=query_instance.detach().cpu().numpy(),
                        original_latent_vector=z_init_single.detach().cpu().numpy(),
                        confidence=best_confidence
                    )
                    
                    pbar.update(1)
                    
        return adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences

    def _compute_adversarial_loss(self, z, z_init, output, original_class):
        """
        Compute the adversarial loss for PGD attack.
        
        Args:
            z: Current latent vector
            z_init: Original latent vector
            output: Model output for the adversarial example
            original_class: Original class of the input
            
        Returns:
            Loss value for untargeted attack
        """
        # Get the output for the current class
        output = output[0]  # Get first item from batch
        
        # Simple cross-entropy loss for untargeted attack
        # For untargeted attacks, we want to minimize the probability of the original class
        loss = output[original_class]
        
        # Return the loss (positive because we want to minimize it with gradient descent)
        return loss
    
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
            'max_linf_distance': linf_distances.max().item()
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
    
    # Create the PGD attack
    pgd_attack = VAEPGDAttack(
        ml_model=ml_model,
        dataset="your_dataset",
        model="your_model",
        epsilon=0.1,
        alpha=0.01,
        num_steps=10,
        vae_model=vae_model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load your data
    data = None  # Replace with your actual data
    
    # Generate adversarial examples
    adv_examples, latent_vectors, success_list, original_latents, confidences = pgd_attack.get_adversarial_examples(data)
    
    # Evaluate the attack
    metrics = pgd_attack.evaluate_attack(data, adv_examples)
    print(f"Attack metrics: {metrics}")
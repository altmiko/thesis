import os
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class GANAttack:
    """
    Adversarial attack for tabular data using GAN.
    Based on the approach from Shukla & Banerjee (2023).
    """
    def __init__(self, 
                 ml_model: nn.Module,
                 dataset: str, 
                 model: str,
                 lambda_: float = 0.5, 
                 epsilon: float = 0.1,
                 gan_model: nn.Module = None, 
                 batch_size: int = 16,
                 device: str = "cpu",
                 targeted: bool = False,
                 target_class: Optional[int] = None
                 ):
        """
        Parameters
        ----------
        ml_model: nn.Module
            Target model to attack.
        gan_model: nn.Module
            Trained GAN model for generating adversarial examples.
        dataset: str
            Dataset name.
        model: str
            Model name.
        lambda_: float
            Weight for balancing classification and similarity losses.
        epsilon: float
            Maximum perturbation allowed in latent space.
        batch_size: int
            Batch size for processing.
        device: str
            Device to use for computation.
        targeted: bool
            Whether to perform targeted attacks.
        target_class: Optional[int]
            Target class for targeted attacks (if None, a random class is chosen).
        """
        self._mlmodel = ml_model
        self.gan = gan_model
        self._dataset = dataset
        self._model = model
        
        self._lambda = lambda_
        self._epsilon = epsilon
        self._batch_size = batch_size
        self.device = device
        self._targeted = targeted
        self._target_class = target_class
        self._attack = "gan_latent"

    def get_adversarial_examples(
        self, 
        factuals: torch.Tensor, 
        folder: str = "adversarial_examples"
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[np.ndarray], List[float]]:
        """
        Generate adversarial examples for the given factual instances.
        
        Parameters:
        ----------
        factuals: torch.Tensor
            Original instances to generate adversarial examples from.
        folder: str
            Folder to save results.
            
        Returns:
        -------
        Tuple containing:
            - List of adversarial examples
            - List of latent vectors
            - List of attack success flags
            - List of original latent vectors
            - List of confidence scores
        """
        # Create folder structure
        sample_num = factuals.shape[0]
        folder = os.path.join(folder, os.path.join(os.path.join(f"{self._dataset}", f"{self._model}"), f"{self._attack}"))
        folder = os.path.join(folder, f"Try_{sample_num}_inputs")
        folder = os.path.join(folder, f"_lambda_{self._lambda}_epsilon_{self._epsilon}")
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            print(f"Folder {folder} already exists. Overwriting...")

        # Generate adversarial examples
        adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences = self._generate_adversarial_examples(factuals, folder)
        
        # Print statistics
        total_successful_attacks = sum(attack_success_list)
        print(f"Total successful adversarial examples: {total_successful_attacks}/{len(factuals)}")
        print(f"Adversarial examples confidence: Mean - {np.mean(confidences)}, Max - {np.max(confidences)}, Min - {np.min(confidences)}")

        return adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences
        
    def _save_npy_files(self, idx: int, folder: str, adversarial_example: np.ndarray, latent_vector: np.ndarray, 
                        attack_success: bool, original_instance: np.ndarray, original_latent_vector: np.ndarray,
                        confidence: float, scale: float):
        """
        Save data for a single factual as .npy files.
        
        Parameters:
        ----------
        idx: int
            Index of the factual.
        folder: str
            Folder to save the files.
        adversarial_example: np.ndarray
            Generated adversarial example.
        latent_vector: np.ndarray
            Latent vector of the adversarial example.
        attack_success: bool
            Whether the attack was successful.
        original_instance: np.ndarray
            Original factual instance.
        original_latent_vector: np.ndarray
            Original latent vector.
        confidence: float
            Confidence of the adversarial example.
        scale: float
            Perturbation scale used.
        """
        np.save(os.path.join(folder, f"adversarial_example_{idx}.npy"), adversarial_example)
        np.save(os.path.join(folder, f"latent_vector_{idx}.npy"), latent_vector)
        np.save(os.path.join(folder, f"attack_success_{idx}.npy"), np.array(attack_success, dtype=bool))
        np.save(os.path.join(folder, f"original_instance_{idx}.npy"), original_instance)
        np.save(os.path.join(folder, f"original_latent_vector_{idx}.npy"), original_latent_vector)
        np.save(os.path.join(folder, f"confidence_{idx}.npy"), np.array(confidence, dtype=float))
        np.save(os.path.join(folder, f"scale_{idx}.npy"), np.array(scale, dtype=float))

    def _generate_adversarial_examples(
        self, factuals: torch.Tensor, folder: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[np.ndarray], List[float]]:
        """
        Internal method to generate adversarial examples using the GAN model.
        
        Parameters:
        ----------
        factuals: torch.Tensor
            Original instances to generate adversarial examples from.
        folder: str
            Folder to save results.
            
        Returns:
        -------
        Same as get_adversarial_examples.
        """
        # Prepare data loader
        test_loader = torch.utils.data.DataLoader(
            factuals, batch_size=self._batch_size, shuffle=False
        )

        # Setup result lists
        adversarial_examples = []
        latent_vectors = []
        original_latent_vectors = []
        attack_success_list = []
        confidences = []

        with tqdm(total=len(factuals), desc="Generating Adversarial Examples") as pbar:
            for idx, query_instances in enumerate(test_loader):
                query_instances = query_instances.float().to(self.device)
                batch_size = query_instances.size(0)

                # Get original predictions
                original_outputs = self._mlmodel.forward(query_instances)
                original_classes = torch.argmax(original_outputs, dim=1).detach().cpu().numpy()

                # Set target classes for attacks
                if self._targeted:
                    if self._target_class is not None:
                        target_classes = torch.full((batch_size,), self._target_class, device=self.device, dtype=torch.long)
                    else:
                        # Random target different from original
                        target_classes = torch.randint(0, self.gan.num_classes - 1, (batch_size,), device=self.device)
                        # Adjust targets to ensure they're different from originals
                        for i in range(batch_size):
                            if target_classes[i].item() >= original_classes[i]:
                                target_classes[i] += 1
                else:
                    # For untargeted attacks, aim for any class other than the original
                    target_classes = [(c + 1) % self.gan.num_classes for c in original_classes]
                    target_classes = torch.tensor(target_classes, device=self.device)
                
                                # Generate adversarial examples using the GAN
                with torch.no_grad():
                    # Split data into categorical and numerical parts
                    x_cat = query_instances[:, :self.gan.num_categorical].long()
                    x_num = query_instances[:, self.gan.num_categorical:]
                    
                    # Get original latent vectors
                    z_original = self.gan.generator.encode(x_cat, x_num)
                    
                    # Try different perturbation scales to find adversarial examples
                    perturbation_scales = [0.2, 0.5, 1.0, 1.5, 2.0] # Different scales of epsilon
                    all_batch_results = []
                    
                    for scale in perturbation_scales:
                        # Generate random perturbation
                        noise = torch.randn_like(z_original) * (self._epsilon * scale)
                        z_perturbed = z_original + noise
                        
                        # Generate adversarial examples from perturbed latent vectors
                        x_adv, _, _ = self.gan.generator.decode(z_perturbed)
                        
                        # Get predictions on adversarial examples
                        adv_outputs = self._mlmodel.forward(x_adv)
                        adv_classes = torch.argmax(adv_outputs, dim=1)
                        
                        # Calculate success and confidence for each example
                        for i in range(batch_size):
                            # For targeted attacks, success means matching the target class
                            # For untargeted attacks, success means any class other than original
                            if self._targeted:
                                success = adv_classes[i].item() == target_classes[i].item()
                                conf = F.softmax(adv_outputs[i], dim=0)[target_classes[i]].item()
                            else:
                                success = adv_classes[i].item() != original_classes[i]
                                # For untargeted attacks, confidence is the complement of original class confidence
                                conf = 1.0 - F.softmax(adv_outputs[i], dim=0)[original_classes[i]].item()
                            
                            all_batch_results.append({
                                'instance_idx': idx * self._batch_size + i,
                                'adv_example': x_adv[i].detach().cpu().numpy(),
                                'latent_vector': z_perturbed[i].detach().cpu().numpy(),
                                'original_latent': z_original[i].detach().cpu().numpy(),
                                'success': success,
                                'confidence': conf,
                                'scale': scale
                            })
                    
                    # Group results by instance index
                    instance_results = {}
                    for result in all_batch_results:
                        idx = result['instance_idx']
                        if idx not in instance_results:
                            instance_results[idx] = []
                        instance_results[idx].append(result)
                    
                    # Select best adversarial example for each instance
                    for instance_idx, results in instance_results.items():
                        # First prioritize successful attacks
                        successful_results = [r for r in results if r['success']]
                        
                        if successful_results:
                            # Among successful attacks, choose the one with highest confidence
                            best_result = max(successful_results, key=lambda r: r['confidence'])
                            attack_success = True
                        else:
                            # If no successful attacks, choose the one with highest confidence
                            best_result = max(results, key=lambda r: r['confidence'])
                            attack_success = False
                        
                        # Add to final results
                        adversarial_examples.append(best_result['adv_example'])
                        latent_vectors.append(best_result['latent_vector'])
                        original_latent_vectors.append(best_result['original_latent'])
                        attack_success_list.append(attack_success)
                        confidences.append(best_result['confidence'])
                        
                        # Save individual results
                        self._save_npy_files(
                            idx=instance_idx,
                            folder=folder,
                            adversarial_example=best_result['adv_example'],
                            latent_vector=best_result['latent_vector'],
                            attack_success=attack_success,
                            original_instance=query_instances[instance_idx % self._batch_size].detach().cpu().numpy(),
                            original_latent_vector=best_result['original_latent'],
                            confidence=best_result['confidence'],
                            scale=best_result['scale']
                        )
                        
                pbar.update(batch_size)
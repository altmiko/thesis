import os
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

class VAEGreedySparsityAttack:
    def __init__(self, 
                 ml_model: nn.Module,
                 dataset: str, 
                 model: str,
                 lambda_: float = 0.5, 
                 optimizer: str = "adam", 
                 lr: float = 1e-2,
                 max_iter: int = 1000,
                 kappa: float = 0.0,
                 max_features: int = None,  # Maximum number of features to modify
                 greedy_steps: int = 50,    # Number of iterations for greedy search
                 vae_model: nn.Module = None,
                 batch_size: int = 16,
                 device: str = "cpu",
                 verbose: bool = False      # Control verbosity of output
                 ):
        self._mlmodel = ml_model  # Your model wrapper
        self._dataset = dataset    # Your dataset
        self._model = model

        self._lambda = lambda_
        self._kappa = kappa
        self._optimizer = optimizer
        self._lr = lr
        self._max_iter = max_iter
        
        # Greedy search parameters
        self._max_features = max_features  # If None, will use greedy search to find optimal number
        self._greedy_steps = greedy_steps  # Number of greedy search iterations
        self._verbose = verbose            # Control detailed logging

        self.vae = vae_model
        self.device = device
        self._batch_size = batch_size
        self._attack = "greedy_sparsity"  # Updated attack name
        
        # Statistics collection
        self.stats = {
            "initial_success_count": 0,
            "greedy_success_count": 0,
            "avg_features_modified": 0,
            "total_examples": 0,
            "feature_importance": {}  # Track which features are most commonly modified
        }

    def get_adversarial_examples(
        self, 
        factuals: torch.Tensor, 
        folder: str = "adversarial_examples"
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[np.ndarray], List[float]]:
        # Save the adversarial examples in the folder named by hyperparameters
        sample_num = factuals.shape[0]
        folder = os.path.join(folder, os.path.join(os.path.join(f"{self._dataset}", f"{self._model}"), f"{self._attack}"))
        folder = os.path.join(folder, f"Try_{sample_num}_inputs")
                                                  
        # Handle None value in folder path
        max_features_str = "None" if self._max_features is None else str(self._max_features)
        folder = os.path.join(folder, f"_lambda_{self._lambda}_lr_{self._lr}_max_iter_{self._max_iter}_maxfeatures_{max_features_str}")
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            print(f"Folder {folder} already exists. Overwriting...")

        adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences = self._adversarial_attack_optimization(factuals, folder)
        
        # Print statistics
        total_successful_attacks = sum(attack_success_list)
        
        print("\n" + "="*50)
        print(f"Attack Statistics:")
        print(f"Total successful adversarial examples: {total_successful_attacks}/{len(factuals)} ({total_successful_attacks/len(factuals)*100:.2f}%)")
        print(f"Initial success rate: {self.stats['initial_success_count']}/{len(factuals)} ({self.stats['initial_success_count']/len(factuals)*100:.2f}%)")
        print(f"Additional success from greedy search: {self.stats['greedy_success_count']} examples")
        
        if self.stats['initial_success_count'] > 0:
            print(f"Average features modified in successful attacks: {self.stats['avg_features_modified']/self.stats['initial_success_count']:.2f}")
        
        if len(confidences) > 0:
            print(f"Adversarial examples confidence: Mean - {np.mean(confidences):.4f}, Max - {np.max(confidences):.4f}, Min - {np.min(confidences):.4f}")
        print("="*50)

        return adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences

    def _get_optimizer(self, params: List[torch.Tensor]) -> torch.optim.Optimizer:
        if self._optimizer == "adam":
            return torch.optim.Adam(params, lr=self._lr)
        elif self._optimizer == "rmsprop":
            return torch.optim.RMSprop(params, lr=self._lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self._optimizer}")

    def _compute_loss(self, z, z_init, adv_example, original_class):
        # Forward pass through the model
        output = self._mlmodel.forward(adv_example)[0]

        # Misclassification loss (C&W formulation)
        # Maximize the difference between the original class confidence and the next highest class confidence
        target_class_score = torch.max(torch.cat((output[:original_class], output[original_class+1:])))
        f = torch.clamp(output[original_class] - target_class_score, min=-self._kappa)

        # Regularisation term for perturbation
        perturbation_loss = torch.norm((z - z_init), 2)

        # Combine losses
        loss = perturbation_loss + self._lambda * f

        return loss, perturbation_loss
    
    def _greedy_feature_search(self, original_instance, adv_example, original_class):
        """
        Perform greedy search to minimize the number of feature modifications.
        
        Args:
            original_instance: Original input example
            adv_example: Adversarial example produced by the latent space optimization
            original_class: Original classification class
            
        Returns:
            Tuple of (optimized adversarial example, success flag, confidence, modified_features_count)
        """
        # Convert tensors to numpy for easier manipulation
        original_np = original_instance.cpu().detach().numpy().flatten()
        adv_np = adv_example.cpu().detach().numpy().flatten()
        
        # Ensure both arrays have the same shape
        if original_np.shape != adv_np.shape:
            if self._verbose:
                print(f"Shape mismatch: original {original_np.shape}, adv {adv_np.shape}")
            # Reshape arrays to match
            min_length = min(len(original_np), len(adv_np))
            original_np = original_np[:min_length]
            adv_np = adv_np[:min_length]
        
        # Calculate absolute differences for each feature
        diffs = np.abs(original_np - adv_np)
        
        # Sort features by their modification magnitude (descending)
        sorted_indices = np.argsort(-diffs)
        
        # Initialize a sparse adversarial example with the original instance
        sparse_adv = original_np.copy()
        
        # Try adding features one by one in order of importance
        best_success = False
        best_confidence = 0.0
        best_example = original_np.copy()
        best_count = 0
        
        # Number of search steps to try
        steps_to_try = min(self._greedy_steps, len(sorted_indices))
        
        # If max_features is set, limit our search to that number
        if self._max_features is not None:
            steps_to_try = min(steps_to_try, self._max_features)
            
        # Progress tracking (only show if verbose is enabled)
        if self._verbose:
            iterator = range(1, steps_to_try + 1)
            print(f"Starting greedy search with {steps_to_try} steps...")
        else:
            iterator = range(1, steps_to_try + 1)
            
        for i in iterator:
            # Select top-i features
            for j in range(i):
                idx = sorted_indices[j]
                sparse_adv[idx] = adv_np[idx]
                
                # Track feature importance for statistics
                feature_idx = int(idx)
                if feature_idx not in self.stats["feature_importance"]:
                    self.stats["feature_importance"][feature_idx] = 0
                self.stats["feature_importance"][feature_idx] += 1
            
            # Reshape to match original instance shape for model input
            sparse_adv_reshaped = sparse_adv.reshape(original_instance.shape)
            
            # Convert to tensor for model evaluation
            sparse_adv_tensor = torch.tensor(sparse_adv_reshaped, device=self.device).float().unsqueeze(0)
            
            # Get model prediction
            output = self._mlmodel.forward(sparse_adv_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = 1 - torch.softmax(output, dim=1)[0, original_class].item()
            
            # Check if the modification is successful
            if pred_class != original_class:
                # Found the minimum set of features needed
                best_success = True
                best_example = sparse_adv_reshaped
                best_confidence = confidence
                best_count = i
                
                # If we only care about minimality, we can break here
                if self._max_features is None:
                    break
        
        if self._verbose:
            print(f"Greedy search result: modified {best_count} features, success: {best_success}")
        
        return best_example, best_success, best_confidence, best_count

    def _adversarial_attack_optimization(
        self, df_fact: torch.Tensor, folder: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[np.ndarray], List[float]]:

        # Prepare data for optimization steps
        test_loader = torch.utils.data.DataLoader(
            df_fact, batch_size=self._batch_size, shuffle=False
        )

        adversarial_examples = []
        latent_vectors = []
        original_latent_vectors = []
        attack_success_list = []
        confidences = []
        
        # Track our success counts for statistics
        self.stats["total_examples"] = len(df_fact)
        self.stats["initial_success_count"] = 0
        self.stats["greedy_success_count"] = 0
        self.stats["avg_features_modified"] = 0

        self._convergence_threshold = 1e-5

        with tqdm(total=len(df_fact), desc="Generating Adversarial Examples") as pbar:
            for idx, query_instances in enumerate(test_loader):
                query_instances = query_instances.float().to(self.device)

                # Get the original class of the instances
                original_classes = torch.argmax(self._mlmodel.forward(query_instances), dim=1).detach().cpu().numpy()

                # Split x_cat and x_num from query_instances
                x_cat = query_instances[:, :self.vae.num_categorical].long()
                x_num = query_instances[:, self.vae.num_categorical:]

                # Get initial latent representation
                z, _ = self.vae.encode(x_cat, x_num)
                z_init = z.clone()  # Store initial z for distance calculation
                z = z.clone().detach().requires_grad_(True)

                for i in range(query_instances.size(0)):  # Iterate over instances in the batch
                    # Extract the instance and its original latent vector
                    z_single = z[i].clone().detach().requires_grad_(True)
                    z_init_single = z_init[i]
                    original_class_single = original_classes[i]
                    original_instance_single = query_instances[i].clone()
                    
                    # Define optimizer for the current instance
                    optim = self._get_optimizer([z_single])

                    # PHASE 1: Standard latent space optimization
                    candidate_adversarial_examples = []
                    candidate_latent_vectors = []
                    candidate_confidences = []
                    candidate_distances = []
                    candidate_misclassification = []
                    
                    for _ in range(self._max_iter):  # Iterate over optimization steps
                        # Decode features
                        adv_example, _, _ = self.vae.decode(z_single.unsqueeze(0))
                        output = self._mlmodel.forward(adv_example)  # Get logits directly

                        # Compute loss
                        loss, distance = self._compute_loss(z_single, z_init_single, adv_example, original_class_single)

                        # Compute confidence
                        confidence_not_original = 1 - torch.softmax(output, dim=1)[0, original_class_single].item()

                        # Backpropagation and optimizer step
                        loss.backward(retain_graph=True)
                        optim.step()
                        optim.zero_grad()

                        # Convergence check
                        if len(candidate_distances) > 1:
                            if abs(candidate_distances[-1] - candidate_distances[-2]) < self._convergence_threshold:
                                break

                        # Store results
                        candidate_adversarial_examples.append(adv_example.detach().cpu().numpy())
                        candidate_latent_vectors.append(z_single.detach().cpu().numpy())
                        candidate_distances.append(distance.detach().cpu().numpy())
                        candidate_confidences.append(confidence_not_original)
                        candidate_misclassification.append((torch.argmax(output, dim=1).item() != original_class_single))

                    # Choose the best adversarial example from Phase 1
                    if any(candidate_misclassification):
                        true_indices = [j for j, misclassified in enumerate(candidate_misclassification) if misclassified]
                        true_distances = [candidate_distances[j] for j in true_indices]
                        best_index = true_indices[np.argmin(true_distances)]
                        initial_adv_example = candidate_adversarial_examples[best_index]
                        initial_latent_vector = candidate_latent_vectors[best_index]
                        initial_success = True
                        initial_confidence = candidate_confidences[best_index]
                        
                        # Ensure consistent shape with original instance
                        initial_adv_example = initial_adv_example.reshape(query_instances[i].cpu().numpy().shape)
                        
                        # Update our statistics
                        self.stats["initial_success_count"] += 1
                        
                        if self._verbose:
                            pbar.write(f"Initial adversarial example found for instance {idx * self._batch_size + i}.")
                    else:
                        # If no misclassification, select by confidence of not being the original class
                        if candidate_confidences:
                            confidence_idx = np.argmax(candidate_confidences)
                            initial_adv_example = candidate_adversarial_examples[confidence_idx]
                            initial_latent_vector = candidate_latent_vectors[confidence_idx]
                            initial_confidence = candidate_confidences[confidence_idx]
                            
                            # Ensure consistent shape with original instance
                            initial_adv_example = initial_adv_example.reshape(query_instances[i].cpu().numpy().shape)
                        else:
                            initial_adv_example = query_instances[i].cpu().numpy()
                            initial_latent_vector = z[i].detach().cpu().numpy()
                            initial_confidence = 0.0
                        initial_success = False
                        
                        if self._verbose:
                            pbar.write(f"No initial adversarial example found for instance {idx * self._batch_size + i}.")
                    
                    # PHASE 2: Greedy feature selection to minimize changes
                    if initial_success:
                        adv_tensor = torch.tensor(initial_adv_example, device=self.device).float()
                        
                        # Use greedy search to minimize feature changes
                        optimized_adv, optimized_success, optimized_confidence, num_features = self._greedy_feature_search(
                            original_instance_single, 
                            adv_tensor,
                            original_class_single
                        )
                        
                        # Update statistics on features modified
                        self.stats["avg_features_modified"] += num_features
                        
                        # If greedy search was successful, use the optimized example
                        if optimized_success:
                            adversarial_example = optimized_adv
                            attack_success = True
                            confidence = optimized_confidence
                            if self._verbose:
                                pbar.write(f"Greedy search successful with {num_features} features modified.")
                        else:
                            # Fall back to the initial adversarial example
                            adversarial_example = initial_adv_example
                            attack_success = initial_success
                            confidence = initial_confidence
                            self.stats["greedy_success_count"] += 1
                            if self._verbose:
                                pbar.write(f"Greedy search unsuccessful, using initial adversarial example.")
                    else:
                        # No initial success, just use the best we found
                        adversarial_example = initial_adv_example
                        attack_success = False
                        confidence = initial_confidence
                    
                    # Ensure adversarial_example has the same shape as factuals
                    original_shape = query_instances[i].cpu().numpy().shape
                    if adversarial_example.shape != original_shape:
                        adversarial_example = adversarial_example.reshape(original_shape)
                    
                    original_latent_vector = z_init[i].detach().cpu().numpy()
                    latent_vector = initial_latent_vector  # Keep the same latent vector from Phase 1

                    # Append results
                    adversarial_examples.append(adversarial_example)
                    latent_vectors.append(latent_vector)
                    attack_success_list.append(attack_success)
                    original_latent_vectors.append(original_latent_vector)
                    confidences.append(confidence)

                    # Save results for the current factual
                    self._save_npy_files(
                        idx=idx * self._batch_size + i,
                        folder=folder,
                        adversarial_example=adversarial_example,
                        latent_vector=latent_vector,
                        attack_success=attack_success,
                        original_instance=query_instances[i].detach().cpu().numpy(),
                        original_latent_vector=original_latent_vector,
                        confidence=confidence
                    )
                    pbar.update(1)
        
        return adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences

    def _save_npy_files(self, idx: int, folder: str, adversarial_example: np.ndarray, latent_vector: np.ndarray, 
                        attack_success: bool, original_instance: np.ndarray, original_latent_vector: np.ndarray,
                        confidence: float):
        """
        Save data for a single factual as .npy files.
        
        :param idx: Index of the factual.
        :param folder: Folder to save the files.
        :param adversarial_example: Generated adversarial example.
        :param latent_vector: Latent vector of the adversarial example.
        :param attack_success: Whether the attack was successful.
        :param original_instance: Original factual instance.
        :param original_latent_vector: Original latent vector.
        :param confidence: Confidence of the chosen adversarial example.
        """

        np.save(os.path.join(folder, f"adversarial_example_{idx}.npy"), adversarial_example)
        np.save(os.path.join(folder, f"latent_vector_{idx}.npy"), latent_vector)
        np.save(os.path.join(folder, f"attack_success_{idx}.npy"), np.array(attack_success, dtype=bool))
        np.save(os.path.join(folder, f"original_instance_{idx}.npy"), original_instance)
        np.save(os.path.join(folder, f"original_latent_vector_{idx}.npy"), original_latent_vector)
        np.save(os.path.join(folder, f"confidence_{idx}.npy"), np.array(confidence, dtype=float))
        
    def get_feature_importance(self, top_n=10):
        """
        Return the most important features for successful attacks.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature_index, count) tuples
        """
        if not self.stats["feature_importance"]:
            return []
            
        # Sort features by importance count
        sorted_features = sorted(self.stats["feature_importance"].items(), 
                                 key=lambda x: x[1], reverse=True)
        
        # Return top N features
        return sorted_features[:top_n]


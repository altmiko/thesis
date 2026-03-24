import os
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

class VAEAttack:
    def __init__(self, 
                 ml_model: nn.Module,
                 dataset: str, 
                 model: str,
                 lambda_: float = 0.5, 
                 optimizer: str = "adam", 
                 lr: float = 1e-2,
                 max_iter: int = 1000,
                 epsilon: float = 0.1,
                 kappa: float = 0.0,
                 vae_model: nn.Module = None,
                 batch_size: int = 16,
                 device: str = "cpu"
                 ):
        self._mlmodel = ml_model  # Your model wrapper
        self._dataset = dataset  # Your dataset
        self._model = model

        self._lambda = lambda_
        self._kappa = kappa
        self._optimizer = optimizer
        self._lr = lr
        self._max_iter = max_iter
        self._epsilon = epsilon

        self.vae = vae_model
        self.device = device
        self._batch_size = batch_size
        self._attack = "baseline"

    def get_adversarial_examples(
        self, 
        factuals: torch.Tensor, 
        folder: str = "adversarial_examples"
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[np.ndarray]]:

        # Save the adversarial examples in the folder named by hyperparameters
        sample_num = factuals.shape[0]
        folder = os.path.join(folder, os.path.join(os.path.join(f"{self._dataset}", f"{self._model}"), f"{self._attack}"))
        folder = os.path.join(folder, f"Try_{sample_num}_inputs")
                                                  
        folder = os.path.join(folder, f"_lambda_{self._lambda}_lr_{self._lr}_max_iter_{self._max_iter}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            print(f"Folder {folder} already exists. Overwriting...")


        adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences = self._adversarial_attack_optimization(factuals, folder)
        
        total_successful_attacks = sum(attack_success_list)
        print(f"Total successful adversarial examples: {total_successful_attacks}/{len(factuals)}")

        print(f"Adversarial examples confidence: Mean - {np.mean(confidences)}, Max - {np.max(confidences)}, Min - {np.min(confidences)}")

        return adversarial_examples, latent_vectors, attack_success_list, original_latent_vectors, confidences

    def _get_optimizer(self, params: List[torch.Tensor]) -> torch.optim.Optimizer:
        if self._optimizer == "adam":
            return torch.optim.Adam(params, lr=self._lr)
        elif self._optimizer == "rmsprop":
            return torch.optim.RMSprop(params, lr=self._lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self._optimizer}")


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

        self._convergence_threshold = 1e-5

        with tqdm(total=len(df_fact), desc="Generating Adversarial Examples") as pbar:
            for idx, query_instances in enumerate(test_loader):
                query_instances = query_instances.float().to(self.device)

                # Get the original class of the instances
                original_classes = torch.argmax(self._mlmodel.forward(query_instances), dim=1).detach().cpu().numpy()  # Use logits directly

                # split x_cat and x_num from query_instances
                x_cat = query_instances[:, :self.vae.num_categorical].long()
                x_num = query_instances[:, self.vae.num_categorical:]

                # Get initial latent representation
                z, _ = self.vae.encode(x_cat, x_num)  # z: Latent vector, _: Ignored additional output
                z_init = z.clone()  # Store initial z for distance calculation
                z = z.clone().detach().requires_grad_(True)

                # Define optimizer
                optim = self._get_optimizer([z])

                candidate_adversarial_examples = [[] for _ in range(query_instances.size(0))]  # For each instance
                candidate_latent_vectors = [[] for _ in range(query_instances.size(0))]
                candidate_confidences = [[] for _ in range(query_instances.size(0))]  # Confidence of not being original class
                candidate_distances = [[] for _ in range(query_instances.size(0))]
                candidate_misclassification = [[] for _ in range(query_instances.size(0))]

                for i in range(query_instances.size(0)):  # Iterate over instances in the batch
                    # Extract the instance and its original latent vector
                    z_single = z[i].clone().detach().requires_grad_(True)
                    z_init_single = z_init[i]
                    original_class_single = original_classes[i]
                    
                    # Define optimizer for the current instance
                    optim = self._get_optimizer([z_single])

                    # Compute epsilon for the current instance
                    # epsilon = self._epsilon * torch.norm(z_init_single, p=2)
                    # print(f"epsilon: {epsilon}")

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

                        # if torch.norm(z_single - z_init_single, p=float('inf')) > self._epsilon:
                        #     print(f"Exceeded epsilon: {torch.norm(z_single - z_init_single, p=float('inf'))}")

                        # z_single.data = torch.clamp(z_single, min=z_init_single - self._epsilon, max=z_init_single + self._epsilon)

                        # convergence check
                        if len(candidate_distances[i]) > 1:
                            if abs(candidate_distances[i][-1] - candidate_distances[i][-2]) < self._convergence_threshold:
                                break

                        # Store results
                        candidate_adversarial_examples[i].append(adv_example.detach().cpu().numpy())
                        candidate_latent_vectors[i].append(z_single.detach().cpu().numpy())
                        candidate_distances[i].append(distance.detach().cpu().numpy())
                        candidate_confidences[i].append(confidence_not_original)
                        candidate_misclassification[i].append((torch.argmax(output, dim=1).item() != original_class_single))

                    # Choose the best adversarial example for each instance
                    if any(candidate_misclassification[i]):
                        true_indices = [j for j, misclassified in enumerate(candidate_misclassification[i]) if misclassified]
                        true_distances = [candidate_distances[i][j] for j in true_indices]
                        best_index = true_indices[np.argmin(true_distances)]
                        adversarial_example = candidate_adversarial_examples[i][best_index]
                        latent_vector = candidate_latent_vectors[i][best_index]
                        attack_success = True
                        confidence = candidate_confidences[i][best_index]
                    else:
                        # If no misclassification, select by confidence of not being the original class
                        if candidate_confidences[i]:
                            confidence_idx = np.argmax(candidate_confidences[i])
                            adversarial_example = candidate_adversarial_examples[i][confidence_idx]
                            latent_vector = candidate_latent_vectors[i][confidence_idx]
                            confidence = candidate_confidences[i][confidence_idx]
                        else:
                            adversarial_example = query_instances[i].detach().cpu().numpy()
                            latent_vector = z[i].detach().cpu().numpy()
                            confidence = 0.0
                        attack_success = False

                    original_latent_vector = z_init[i].detach().cpu().numpy()

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


    # def _compute_loss(self, z, z_init, adv_example, original_class):
    #     output = self._mlmodel.forward(adv_example)[0]

    #     loss1 = output[original_class]  # Confidence for the original class
    #     loss2 = torch.norm((z - z_init), 2)

    #     loss = self._lambda * loss1 + loss2

    #     return loss, loss2

    # def _compute_loss(self, z, z_init, adv_example, original_class):
    #     output = self._mlmodel.predict_proba(adv_example)[0]

    #     # First Term: Encourage classification into a class other than the original class
    #     loss1 = output[original_class] - torch.max(output[torch.arange(output.size(0)) != original_class])

    #     # Second Term: Distance loss in latent space
    #     loss2 = torch.norm((z - z_init), 2)

    #     return loss1 + self._lambda * loss2
    

    # def _compute_loss(self, z, z_init, adv_example, original_class):
    #     """
    #     Computes the loss for untargeted adversarial attacks.

    #     Args:
    #         z (torch.Tensor): Latent representation of the adversarial example.
    #         z_init (torch.Tensor): Latent representation of the original input.
    #         adv_example (torch.Tensor): Adversarial example.
    #         original_class (int): The original class of the input.

    #     Returns:
    #         torch.Tensor: The computed loss value.
    #     """
    #     output = self._mlmodel.predict_proba(adv_example)[0]  # Model predictions (softmax probabilities)

    #     # Classification loss: Maximise the confidence of classes other than the original class
    #     max_other_class_confidence = max(output[i] for i in range(len(output)) if i != original_class)
    #     loss1 = max_other_class_confidence - output[original_class]  # Confidence difference

    #     # Confidence margin (optional, for more control)
    #     margin = -self._k if hasattr(self, "_k") else 0
    #     loss1 = torch.clamp(loss1, min=margin)  # Ensure misclassification margin

    #     # Distance loss in latent space
    #     loss2 = torch.norm((z - z_init), p=2)  # L2 distance in latent space

    #     # Combine the losses with a weighting factor
    #     return self._lambda * loss1 +  loss2


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

import itertools
import numpy as np
import os
import json
from typing import Callable, Dict, List, Tuple, Any, Optional
from scipy.linalg import inv
from scipy.stats import chi2

import torch
import torch.nn as nn


class AttackFactory:
    """Factory class to create appropriate attack instances based on attack type."""
    
    @staticmethod
    def create_attack(attack_type: str, attack_class: classmethod, ml_model: nn.Module, 
                     dataset: str, model: str, params: Dict[str, Any], 
                     vae_model: nn.Module, batch_size: int, device: str) -> Any:
        """
        Create an attack instance based on the attack type.
        
        Args:
            attack_type: Type of the attack ('baseline', 'pgd', etc.)
            attack_class: The attack class to instantiate
            ml_model: The machine learning model to attack
            dataset: Dataset name
            model: Model name
            params: Attack parameters
            vae_model: VAE model
            batch_size: Batch size
            device: Device to run on
            
        Returns:
            An instance of the attack
        """
        if attack_type == 'baseline':
            return attack_class(
                ml_model=ml_model,
                dataset=dataset,
                model=model,
                lambda_=params['_lambda'],
                optimizer=params['optimizer'],
                lr=params['lr'],
                max_iter=params['max_iter'],
                kappa=params['kappa'],
                vae_model=vae_model,
                batch_size=batch_size,
                device=device
            )
        elif attack_type == 'pgd':
            return attack_class(
                ml_model=ml_model,
                dataset=dataset,
                model=model,
                epsilon=params['epsilon'],
                alpha=params['alpha'],
                num_steps=params['num_steps'],
                vae_model=vae_model,
                batch_size=batch_size,
                device=device
            )
        elif attack_type == 'deltaz':
            return attack_class(
                ml_model=ml_model,
                dataset=dataset,
                model=model,
                lambda_=params['_lambda'],
                p_norm=params['p_norm'],
                lr=params['lr'],
                max_iter=params['max_iter'],
                vae_model=vae_model,
                batch_size=batch_size,
                device=device
            )
        elif attack_type == 'sparsity':
            return attack_class(
                ml_model=ml_model,
                dataset=dataset,
                model=model,
                lambda_=params['_lambda'],
                lambda_sparsity=params['lambda_sparsity'],
                optimizer=params['optimizer'],
                lr=params['lr'],
                max_iter=params['max_iter'],
                kappa=params['kappa'],
                gamma=params['gamma'],
                vae_model=vae_model,
                batch_size=batch_size,
                device=device
            )
        # In AttackFactory.create_attack() method
        elif attack_type == 'greedy_sparsity':
            return attack_class(
                ml_model=ml_model,
                dataset=dataset,
                model=model,
                lambda_=params['_lambda'],
                optimizer=params['optimizer'],
                lr=params['lr'],
                max_iter=params['max_iter'],
                kappa=params['kappa'],
                max_features=params.get('max_features'),  # Optional parameter
                greedy_steps=params.get('greedy_steps', 50),  # Default to 50 if not provided
                vae_model=vae_model,
                batch_size=batch_size,
                device=device,
                verbose=False
            )
        elif attack_type == 'sparsity_l1':
            return attack_class(
                ml_model=ml_model,
                dataset=dataset,
                model=model,
                lambda_=params['_lambda'],
                lambda_sparsity=params['lambda_sparsity'],
                optimizer=params['optimizer'],
                lr=params['lr'],
                max_iter=params['max_iter'],
                kappa=params['kappa'],
                vae_model=vae_model,
                batch_size=batch_size,
                device=device
            )
        else:
            raise ValueError(f"Attack type {attack_type} not supported.")


class PathManager:
    """Manages paths for storing and loading attack results."""
    
    @staticmethod
    def get_base_folder(root_folder: str, dataset: str, model: str, attack: str) -> str:
        """
        Get the base folder path for storing results.
        
        Args:
            root_folder: Root folder for all results
            dataset: Dataset name
            model: Model name
            attack: Attack name
            
        Returns:
            Base folder path
        """
        data_model_folder = os.path.join(root_folder, dataset, model)
        return os.path.join(data_model_folder, attack)
    
    @staticmethod
    def get_sample_folder(base_folder: str, sample_num: int) -> str:
        """
        Get the folder path for a specific sample size.
        
        Args:
            base_folder: Base folder path
            sample_num: Number of samples
            
        Returns:
            Sample folder path
        """
        output_folder = os.path.join(base_folder, f"Try_{sample_num}_inputs")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder
    
    @staticmethod
    def get_result_file(folder: str, attack: str, params: Dict[str, Any]) -> str:
        """
        Get the result file path for specific attack parameters.
        
        Args:
            folder: Folder to store the result
            attack: Attack type
            params: Attack parameters
            
        Returns:
            Result file path
        """
        if attack == 'baseline':
            return os.path.join(
                folder,
                f"lambda_{params['_lambda']}_lr_{params['lr']}_max_iter_{params['max_iter']}_results.json"
            )
        elif attack == 'pgd':
            return os.path.join(
                folder,
                f"epsilon_{params['epsilon']}_alpha_{params['alpha']}_steps_{params['num_steps']}_results.json"
            )
        elif attack == 'deltaz':
            return os.path.join(
                folder,
                f"_lambda_{params['_lambda']}_p_{params['p_norm']}_lr_{params['lr']}_max_iter_{params['max_iter']}_results.json"
            )
        elif attack == 'sparsity':
            return os.path.join(
                folder,
                (
                    f"lambda_{params['_lambda']}_sparsity_{params['lambda_sparsity']}_lr_{params['lr']}_max_iter_{params['max_iter']}"
                    f"_results.json"
                )
            )
        elif attack == 'greedy_sparsity':
            max_features_str = f"maxfeatures_{params.get('max_features', 'None')}"
            return os.path.join(
                folder,
                f"lambda_{params['_lambda']}_lr_{params['lr']}_max_iter_{params['max_iter']}_{max_features_str}_results.json"
            )
        elif attack == 'sparsity_l1':
            return os.path.join(
                folder,
                (
                    f"lambda_{params['_lambda']}_sparsity_{params['lambda_sparsity']}_lr_{params['lr']}_max_iter_{params['max_iter']}"
                    f"_results.json"
                )
            )
        else:
            raise ValueError(f"Attack type {attack} not supported.")
    
    @staticmethod
    def get_attack_folder(base_folder: str, attack: str, params: Dict[str, Any]) -> str:
        """
        Get the folder path for storing adversarial examples.
        
        Args:
            base_folder: Base folder path
            attack: Attack type
            params: Attack parameters
            
        Returns:
            Attack folder path
        """
        output_folder = os.path.join(base_folder, f"Try_{params['sample_num']}_inputs")
        
        if attack == 'baseline':
            return os.path.join(
                output_folder,
                f"_lambda_{params['_lambda']}_lr_{params['lr']}_max_iter_{params['max_iter']}"
            )
        elif attack == 'pgd':
            return os.path.join(
                output_folder,
                f"_epsilon_{params['epsilon']}_alpha_{params['alpha']}_steps_{params['num_steps']}"
            )
        elif attack == 'deltaz':
            return os.path.join(
                output_folder,
                f"_lambda_{params['_lambda']}_p_{params['p_norm']}_lr_{params['lr']}_max_iter_{params['max_iter']}"
            )
        elif attack == 'sparsity':
            return os.path.join(
                output_folder,
                f"_lambda_{params['_lambda']}_sparsity_{params['lambda_sparsity']}_lr_{params['lr']}_max_iter_{params['max_iter']}"
            )
        elif attack == 'greedy_sparsity':
            max_features_str = f"maxfeatures_{params.get('max_features', 'None')}"
            return os.path.join(
                output_folder,
                f"_lambda_{params['_lambda']}_lr_{params['lr']}_max_iter_{params['max_iter']}_{max_features_str}"
            )
        elif attack == 'sparsity_l1':
            return os.path.join(
                output_folder,
                f"_lambda_{params['_lambda']}_sparsity_{params['lambda_sparsity']}_lr_{params['lr']}_max_iter_{params['max_iter']}"
            )
        else:
            raise ValueError(f"Attack type {attack} not supported.")


class GridSearch:
    def __init__(
        self,
        attack_type: classmethod,
        ml_model: nn.Module,
        data: str,
        model: str,
        attack: str,
        vae_model: nn.Module,
        factuals: torch.Tensor,
        train_data: torch.Tensor,
        parameter_grid: Dict[str, List],
        evaluation_metric: Callable,
        batch_size: int = 16,
        device: str = "cpu",
        folder: str = "results",
        load_existing: bool = False,
        override_sample_num: int = None
    ):
        """
        Initialises the GridSearch class for hyperparameter tuning.

        Args:
            attack_type: The attack class to instantiate.
            ml_model: The machine learning model to attack.
            data: Dataset name.
            model: Model name.
            attack: Attack type ('baseline', 'pgd', etc.).
            vae_model: Pre-trained Variational Autoencoder for latent representation.
            factuals: The instances to attack (input data).
            train_data: Training data for Mahalanobis distance calculation.
            parameter_grid: Dictionary containing parameter names and their possible values.
            evaluation_metric: Function to evaluate performance.
            batch_size: Batch size for attacks.
            device: The device to use for computation ('cpu' or 'cuda').
            folder: Folder to store the results.
            load_existing: Whether to load existing results instead of recomputing.
        """
        self.attack_type = attack_type
        self.ml_model = ml_model
        self.data = data
        self.model = model
        self.attack = attack
        self.vae_model = vae_model
        self.factuals = factuals
        self.train_data = train_data
        self.parameter_grid = parameter_grid
        self.evaluation_metric = evaluation_metric
        self.batch_size = batch_size
        self.device = device
        self.folder = folder
        self.load_existing = load_existing
        self.override_sample_num = override_sample_num

        # Create base folder
        self.base_folder = PathManager.get_base_folder(
            self.folder, self.data, self.model, self.attack
        )
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)

    def run(self) -> Tuple[Dict[str, any], Dict[str, float]]:
        """
        Runs the grid search for hyperparameter tuning.

        Returns:
            A tuple containing the best hyperparameters and their associated metrics.
        """
        best_params = None
        best_metrics = None
        best_score = float('-inf')

        # Generate all combinations of parameters
        param_combinations = list(itertools.product(*self.parameter_grid.values()))
        param_names = list(self.parameter_grid.keys())

        # Prepare Mahalanobis distance parameters
        md_params = self.prepare_md()

        # List to store overall results
        overall_results = []

        for i, combination in enumerate(param_combinations):
            print("------------------------------------------------------------")
            print(f"Running combination {i+1}/{len(param_combinations)}\n")

            # Create a dictionary of current parameters
            params = dict(zip(param_names, combination))
            print(f"Testing parameters: {params}")

            tmp_num = params['sample_num']
            tmp_params = params.copy()
            if self.override_sample_num is not None:
                tmp_num = self.override_sample_num
                tmp_params['sample_num'] = self.override_sample_num

            # Get the sample folder for this run
            sample_folder = PathManager.get_sample_folder(
                self.base_folder, tmp_num
            )
        
            # Get the result file path for this run
            result_file = PathManager.get_result_file(
                sample_folder, self.attack, params
            )
            
            # Check if results already exist and should be loaded
            if os.path.exists(result_file) and not self.load_existing:
                print(f"Results for parameters {params} already exist. Skipping execution.")
                with open(result_file, "r") as f:
                    existing_result = json.load(f)
                    overall_results.append(existing_result)
                    continue
            
            # Get attack results (either by loading existing or running attack)
            if self.load_existing:
                try:
                    # Try to load existing results
                    results = self._load_existing_results(tmp_params)
                    print(f"Successfully loaded existing results for parameters: {params}")
                except (FileNotFoundError, ValueError) as e:
                    # If files don't exist or there's an error loading them, fall back to running the attack
                    print(f"Could not load existing results: {e}")
                    print(f"Running attack instead for parameters: {params}")
                    results = self._run_attack(params)
            else:
                results = self._run_attack(params)
                
            # Extract results
            adversarial_examples, latent_vectors, success_list, original_latent_vectors, confidences = results
                
            # Evaluate the performance metrics
            metrics = self.evaluation_metric(
                self.factuals,
                adversarial_examples,
                latent_vectors,
                success_list,
                original_latent_vectors,
                confidences,
                md_params
            )
            misclassification_rate = metrics[0]

            # Store the individual result
            result = self._create_result_dict(params, metrics)
            overall_results.append(result)

            print(f"Results: \n{json.dumps(result, indent=4)}\n")

            # Save the result for the current parameter set
            with open(result_file, "w") as f:
                json.dump(result, f)

            # Update the best parameters based on misclassification rate
            if misclassification_rate > best_score:
                best_score = misclassification_rate
                best_params = params
                best_metrics = {
                    'success_rate': metrics[0],
                    'mean_l2_distance': metrics[1],
                    'mean_linf_distance': metrics[2],
                }

        # Save overall results to a single file
        overall_results_file = os.path.join(sample_folder, "overall_results.json")
        with open(overall_results_file, "w") as f:
            json.dump(overall_results, f, indent=4)

        return best_params, best_metrics
    
    def _run_attack(self, params: Dict[str, Any]) -> Tuple[List, List, List, List, List]:
        """
        Run the attack with specified parameters.
        
        Args:
            params: Attack parameters
            
        Returns:
            Tuple of (adversarial_examples, latent_vectors, success_list, original_latent_vectors, confidences)
        """
        # Create attack instance
        attack = AttackFactory.create_attack(
            self.attack,
            self.attack_type,
            self.ml_model,
            self.data,
            self.model,
            params,
            self.vae_model,
            self.batch_size,
            self.device
        )
        
        # Run the attack
        return attack.get_adversarial_examples(self.factuals)
    
    def _load_existing_results(self, params: Dict[str, Any]) -> Tuple[List, List, List, List, List]:
        """
        Load existing attack results.
        
        Args:
            params: Attack parameters
            
        Returns:
            Tuple of (adversarial_examples, latent_vectors, success_list, original_latent_vectors, confidences)
        
        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If there are issues with the loaded data
        """
        # Get the attack folder path to check if it exists
        base_folder = PathManager.get_base_folder("adversarial_examples", self.data, self.model, self.attack)
        attack_folder = PathManager.get_attack_folder(base_folder, self.attack, params)
        
        if not os.path.exists(attack_folder):
            raise FileNotFoundError(f"Attack folder does not exist: {attack_folder}")
        
        # Check if at least one file exists in the folder
        sample_file = os.path.join(attack_folder, f"original_instance_0.npy")
        if not os.path.exists(sample_file):
            raise FileNotFoundError(f"No files found in attack folder: {attack_folder}")
        
        # Load datapoints
        original_instances, original_latent_vectors, latent_vectors, adversarial_examples, success_list, confidences = load_datapoints(
            params, self.data, self.model, self.attack
        )
        
        return adversarial_examples, latent_vectors, success_list, original_latent_vectors, confidences
    
    def _create_result_dict(self, params: Dict[str, Any], metrics: Tuple) -> Dict[str, Any]:
        """
        Create a dictionary with the results.
        
        Args:
            params: Attack parameters
            metrics: Evaluation metrics
            
        Returns:
            Dictionary with results
        """
        return {
            "params": params,
            "metrics": {
                "success_rate": float(metrics[0]),
                "mean_l2_distance": float(metrics[1]),
                "mean_linf_distance": float(metrics[2]),
                "perturbed_features_latent": float(metrics[3]),
                "perturbed_features_input": float(metrics[4]),
                "mean_confidence_successful": float(metrics[5]),
                "mean_confidence_unsuccessful": float(metrics[6]),
                "min_confidence_unsuccessful": float(metrics[7]),
                "max_confidence_unsuccessful": float(metrics[8]),
                "mean_md_distance": float(metrics[9]),
                "outlier_rate": float(metrics[10]),
                "outliers": f"{sum(metrics[11])}/{len(metrics[11]) if isinstance(metrics[11], np.ndarray) else 0}",
            }
        }

    def prepare_md(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Prepare parameters for Mahalanobis distance calculation.
        
        Returns:
            Tuple of (mean, inverse covariance matrix, threshold)
        """
        latent_train, _ = self.vae_model.encode(
            self.train_data[:, : self.vae_model.num_categorical].long(), 
            self.train_data[:, self.vae_model.num_categorical :]
        )
        latent_train = latent_train.cpu().detach().numpy()
        mu = np.mean(latent_train, axis=0)
        cov = np.cov(latent_train, rowvar=False)

        # Regularisation for covariance
        lambda_reg = 1e-6
        cov_reg = cov + lambda_reg * np.eye(cov.shape[0])

        # Inverse covariance matrix
        cov_inv = inv(cov_reg)

        # Chi-Squared threshold
        d = latent_train.shape[1]  # Dimensionality of latent space
        alpha = 0.05  # Significance level
        threshold = np.sqrt(chi2.ppf(1 - alpha, df=d))

        return mu, cov_inv, threshold


def mahalanobis(z, mu, cov_inv):
    """
    Calculate the Mahalanobis distance.
    
    Args:
        z: Point
        mu: Mean
        cov_inv: Inverse covariance matrix
        
    Returns:
        Mahalanobis distance
    """
    diff = z - mu
    return np.sqrt(diff.T @ cov_inv @ diff)


def evaluation_metric(
    factuals: torch.Tensor,
    adversarial_examples: List[np.ndarray],
    latent_vectors: List[np.ndarray],
    success_list: List[bool],
    original_latent_vectors: List[np.ndarray],
    confidences: List[np.ndarray],
    md_params: Tuple[np.ndarray, np.ndarray, float]
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, np.ndarray]:
    """
    Evaluates the performance of adversarial attacks.

    Args:
        factuals: List of factual instances.
        adversarial_examples: List of generated adversarial examples.
        latent_vectors: List of latent vectors for adversarial examples.
        success_list: List of booleans indicating whether each attack was successful.
        original_latent_vectors: List of original latent vectors.
        confidences: List of confidence values.
        md_params: Tuple of (mean, inverse covariance matrix, threshold) for Mahalanobis distance.
        
    Returns:
        Tuple of evaluation metrics.
    """
    # Misclassification rate
    misclassification_rate = float(sum(success_list)) / len(success_list)

    adversarial_examples = np.array(adversarial_examples)
    latent_vectors = np.array(latent_vectors)
    original_latent_vectors = np.array(original_latent_vectors)
    confidences = np.array(confidences)
    factuals = factuals.cpu().numpy()

    # Compute L2 and L-inf distances
    l2_distances = [
        np.linalg.norm(latent_vectors[i] - original_latent_vectors[i], ord=2)
        for i, success in enumerate(success_list) if success
    ]
    linf_distances = [
        np.linalg.norm(latent_vectors[i] - original_latent_vectors[i], ord=np.inf)
        for i, success in enumerate(success_list) if success
    ]
    
    # Compute Mahalanobis distances
    mu, cov_inv, threshold = md_params
    md_distances = [
        mahalanobis(latent_vectors[i], mu, cov_inv)
        for i, success in enumerate(success_list) if success
    ]
    outliers = np.array(md_distances) > threshold

    mean_md_distance = np.mean(md_distances) if len(md_distances) > 0 else 0.0
    outlier_rate = np.mean(outliers) if len(outliers) > 0 else 0.0

    # compute mean number of perturbed features in both latent space and input space
    delta = 1e-5
    perturbed_features_latent = [
        np.sum(np.abs(latent_vectors[i] - original_latent_vectors[i]) > delta)
        for i, success in enumerate(success_list) if success
    ]
    perturbed_features_input = [
        np.sum(np.abs(adversarial_examples[i] - factuals[i]) > delta)
        for i, success in enumerate(success_list) if success
    ]

    mean_perturbed_features_latent = np.mean(perturbed_features_latent) if perturbed_features_latent else 0.0
    mean_perturbed_features_input = np.mean(perturbed_features_input) if perturbed_features_input else 0.0

    # Mean distances
    mean_l2_distance = np.mean(l2_distances) if len(l2_distances) > 0 else 0.0
    mean_linf_distance = np.mean(linf_distances) if len(linf_distances) > 0 else 0.0

    # Confidences metrics
    # 1. Mean confidence of the successful attacks
    success_array = np.array(success_list)
    mean_confidence_successful = np.mean(confidences[success_array]) if np.any(success_array) else 0.0
    # 2. Mean confidence of the unsuccessful attacks
    mean_confidence_unsuccessful = np.mean(confidences[~success_array]) if np.any(~success_array) else 0.0
    # 3. Min confidence of the unsuccessful attacks
    min_confidence_unsuccessful = np.min(confidences[~success_array]) if np.any(~success_array) else 0.0
    # 4. Max confidence of the unsuccessful attacks
    max_confidence_unsuccessful = np.max(confidences[~success_array]) if np.any(~success_array) else 0.0

    return (misclassification_rate, mean_l2_distance, mean_linf_distance, 
            mean_perturbed_features_latent, mean_perturbed_features_input,
            mean_confidence_successful, mean_confidence_unsuccessful, 
            min_confidence_unsuccessful, max_confidence_unsuccessful,
            mean_md_distance, outlier_rate, outliers)


def load_datapoints(params, dataset, model, attack, folder="adversarial_examples"):
    """
    Load the dataset and return the factual instances.

    Args:
        params: Dictionary containing the dataset parameters.
        dataset: The dataset to load.
        model: The model name.
        attack: The attack type.
        folder: The folder to load from.
        
    Returns:
        Tuple of (original_instances, original_latent_vectors, latent_vectors, 
                 adversarial_examples, attack_success, confidences)
                 
    Raises:
        FileNotFoundError: If any required file doesn't exist
        ValueError: If there are issues with the loaded data
    """
    # Get the attack folder
    base_folder = PathManager.get_base_folder(folder, dataset, model, attack)
    attack_folder = PathManager.get_attack_folder(base_folder, attack, params)
    
    if not os.path.exists(attack_folder):
        raise FileNotFoundError(f"Attack folder does not exist: {attack_folder}")
    
    original_instances = []
    original_latent_vectors = []
    latent_vectors = []
    adversarial_examples = []
    attack_success = []
    confidences = []

    for i in range(params['sample_num']):
        counter = i

        # Define file paths for all components
        original_instance_file = os.path.join(attack_folder, f"original_instance_{counter}.npy")
        original_latent_vector_file = os.path.join(attack_folder, f"original_latent_vector_{counter}.npy")
        latent_vector_file = os.path.join(attack_folder, f"latent_vector_{counter}.npy")
        adversarial_example_file = os.path.join(attack_folder, f"adversarial_example_{counter}.npy")
        attack_success_file = os.path.join(attack_folder, f"attack_success_{counter}.npy")
        confidence_file = os.path.join(attack_folder, f"confidence_{counter}.npy")
        
        # Check if all required files exist
        required_files = [
            original_instance_file, original_latent_vector_file, 
            latent_vector_file, adversarial_example_file,
            attack_success_file, confidence_file
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        try:
            # Load all components
            original_instance = np.load(original_instance_file)
            original_latent_vector = np.load(original_latent_vector_file)
            latent_vector = np.load(latent_vector_file)
            adversarial_example = np.load(adversarial_example_file)
            success = np.load(attack_success_file)
            confidence = np.load(confidence_file)
            
            # Append to respective lists
            original_instances.append(original_instance)
            original_latent_vectors.append(original_latent_vector)
            latent_vectors.append(latent_vector)
            adversarial_examples.append(adversarial_example)
            attack_success.append(success)
            confidences.append(confidence)
            
        except Exception as e:
            raise ValueError(f"Error loading data for sample {counter}: {str(e)}")

    try:
        # Convert lists to arrays
        original_instances = np.vstack(original_instances)
        original_latent_vectors = np.array(original_latent_vectors)
        latent_vectors = np.array(latent_vectors)
        adversarial_examples = np.array(adversarial_examples)
        attack_success = np.array(attack_success)
        confidences = np.array(confidences)
    except Exception as e:
        raise ValueError(f"Error converting data to arrays: {str(e)}")

    return original_instances, original_latent_vectors, latent_vectors, adversarial_examples, attack_success, confidences


def sample_data_equal_class(data, labels=None, n_samples=1000):
    """
    Sample data with equal number of samples from each class.
    
    Args:
        data: Input data.
        labels: Labels for the data.
        n_samples: Number of samples to return.
        
    Returns:
        Tuple of (sampled_data, sampled_labels)
    """
    np.random.seed(42)
    if labels is None:
        if data.shape[0] > n_samples:
            indices = np.random.choice(data.shape[0], n_samples, replace=False)
            sampled_data = data[indices]
            return sampled_data, None
        return data, None
    
    # Get unique classes and their counts
    unique_classes = np.unique(labels)
    samples_per_class = n_samples // len(unique_classes)
    
    sampled_data_list = []
    sampled_labels_list = []
    
    for class_label in unique_classes:
        class_indices = np.where(labels == class_label)[0]
        if len(class_indices) > samples_per_class:
            selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        else:
            selected_indices = class_indices
            
        sampled_data_list.append(data[selected_indices])
        sampled_labels_list.append(labels[selected_indices])
    
    sampled_data = torch.cat(sampled_data_list)
    sampled_labels = np.concatenate(sampled_labels_list)
    
    return sampled_data, sampled_labels
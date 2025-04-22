import torch
from torch.autograd import grad
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from collections import Counter
import numpy as np
import warnings
from torch.nn import functional as F

class BaseGrad:
    """
    Base class for gradient-based saliency/class activation map generators.
    
    This class provides the foundation for creating saliency maps that highlight
    important regions in input images that contribute to model predictions.
    
    The base implementation uses vanilla gradients to generate the saliency maps.
    """
    def __init__(self, model: torch.nn.Module, *args: Any, **kwargs: Any):
        """
        Initialize the gradient-based saliency map generator.
        
        Args:
            model: Neural network model for which saliency maps will be generated
            device: Device (CPU/GPU) where computations will be performed
        """
        self.model = model

    def generate_maps(self, X: torch.Tensor, class_idx: Optional[List[int]] = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Generate saliency maps for the input images.

        Args:
            X: Input images to generate saliency maps for
            class_idx: Optional class indices to generate saliency maps for.
                       If None, uses the classes predicted by the model.
                       If provided but length doesn't match batch size, defaults to predicted classes.
            *args: Additional positional arguments for derived classes
            **kwargs: Additional keyword arguments for derived classes

        Returns:
            Saliency maps for the input images with shape matching the input tensor dimensions
        """
        self.model.eval()

        # Make sure we track gradients for input
        X.requires_grad_(True)
        
        # Forward pass to get model predictions
        model_output = self.model(X)
        
        # Get predicted classes if class_idx is not provided
        batch_size = X.shape[0]
        if class_idx is None:
            # Use model's predicted class (highest probability)
            _, predicted_classes = torch.max(model_output, dim=1)
            target_classes = predicted_classes
        else:
            # Check if provided class_idx matches batch size
            if len(class_idx) != batch_size:
                warnings.warn(
                    f"Length of class_idx ({len(class_idx)}) does not match batch size ({batch_size}). "
                    "Defaulting to predicted classes."
                )
                _, predicted_classes = torch.max(model_output, dim=1)
                target_classes = predicted_classes
            else:
                target_classes = torch.tensor(class_idx, device=X.device)
        
        # Create a one-hot encoding for the target classes
        one_hot = torch.zeros_like(model_output)
        for i in range(batch_size):
            one_hot[i, target_classes[i]] = 1
            
        # Calculate gradients with respect to input
        model_output.backward(gradient=one_hot)
        
        # Get the gradients
        saliency_maps = X.grad.detach().abs()
        saliency_maps = saliency_maps.mean(dim=1)
        
        # Clean up
        X.requires_grad_(False)
        
        return saliency_maps, target_classes
    


class SmoothGrad:
    """    
    This class applies SmoothGrad technique to any gradient-based attribution method
    by adding Gaussian noise to the input multiple times and averaging the resulting
    saliency maps to reduce visual noise.
    """
    def __init__(
        self, 
        base_grad_class: BaseGrad, 
        std: float = 0.15, 
        n_samples: int = 50, 
        perturb_batch_size: int = 10, 
        *args: Any, 
        **kwargs: Any
    ):
        """
        Initialize the SmoothGrad with a base gradient method.
        
        Args:
            base_grad_class: Instance of a class inheriting from BaseGrad
            std: Standard deviation for Gaussian noise (default: 0.15)
            n_samples: Number of noisy samples to average (default: 50)
            perturb_batch_size: Batch size for processing noise samples (default: 10)
            *args: Additional positional arguments for the base gradient class
            **kwargs: Additional keyword arguments for the base gradient class
        """
        self.base_grad_class = base_grad_class
        self.std = std
        self.n_samples = n_samples
        self.perturb_batch_size = perturb_batch_size

    def generate_maps(
        self, 
        X: torch.Tensor, 
        class_idx: Optional[List[int]] = None, 
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate smoothed saliency maps using SmoothGrad technique.
        
        Args:
            X: Input images to generate saliency maps for
            class_idx: Optional class indices to generate saliency maps for
                      If None, uses the classes predicted by the model
                      If provided but length doesn't match batch size, defaults to predicted classes
            *args: Additional positional arguments for the base class
            **kwargs: Additional keyword arguments for the base class
            
        Returns:
            Tuple containing:
                - Smoothed saliency maps for the input images
                - Target classes used for generating the maps (most frequent class for each image)
        """
        device = X.device
        input_batch_size = X.shape[0]
        accumulated_maps = torch.zeros((input_batch_size,) + X.shape[2:], device=device)
        
        # Dictionary to track predicted classes for each image
        # Keys are image indices, values are Counters of predicted classes
        class_counters: Dict[int, Counter] = {i: Counter() for i in range(input_batch_size)}

        # Get the range of input per image (used to scale the noise)
        # Shape: (batch_size, 1, 1, 1)
        perturb_ranges = (X.max(dim=1, keepdim=True)[0] - X.min(dim=1, keepdim=True)[0]) * self.std

        samples_processed = 0

        while samples_processed < self.n_samples:
            current_batch_size = min(self.perturb_batch_size, self.n_samples - samples_processed)

            # Generate noise and scale by per-image perturb_ranges
            noise = torch.randn((current_batch_size, *X.shape), device=device)  # (B, N, C, H, W)
            noise = noise * perturb_ranges.unsqueeze(0)  # Broadcast along sample dimension

            # Add noise to the input image
            noisy_inputs = X.unsqueeze(0) + noise  # (B, N, C, H, W)
            noisy_inputs = noisy_inputs.view(-1, *X.shape[1:])  # Flatten to (B*N, C, H, W)

            # Compute saliency maps using base gradient method
            saliency_maps, target_classes = self.base_grad_class.generate_maps(noisy_inputs, class_idx, *args, **kwargs)

            # Reshape maps to (batch_size, num_images, H, W)
            saliency_maps = saliency_maps.view(current_batch_size, input_batch_size, *X.shape[2:])
            
            # Count predicted classes for each image
            target_classes_reshaped = target_classes.view(current_batch_size, input_batch_size)
            for sample_idx in range(current_batch_size):
                for img_idx in range(input_batch_size):
                    pred_class = target_classes_reshaped[sample_idx, img_idx].item()
                    class_counters[img_idx][pred_class] += 1

            # Accumulate the saliency maps
            accumulated_maps += saliency_maps.sum(dim=0)

            samples_processed += current_batch_size

        # Average accumulated maps over number of samples
        averaged_maps = accumulated_maps / self.n_samples
        
        # Get the most frequent class for each image
        most_frequent_classes = torch.tensor([
            counter.most_common(1)[0][0] for counter in class_counters.values()
        ], device=device)

        return averaged_maps, most_frequent_classes


class IntegratedGradients(BaseGrad):
    """    
    This class implements the Integrated Gradients technique which computes the path integral
    of gradients along a straight line from a baseline input to the actual input to attribute
    the prediction to input features.
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        n_steps: int = 5, 
        baseline: Optional[Union[torch.Tensor, str]] = "random", 
        *args: Any, 
        **kwargs: Any
    ):
        """
        Initialize the Integrated Gradients method.
        
        Args:
            model: Neural network model for which attributions will be generated
            n_steps: Number of steps for the Riemann approximation of the integral (default: 5)
            baseline: Baseline input for the path integral. Can be 'zero', 'random', or a tensor with same shape as input.
                     If 'zero', uses an all-zero tensor as baseline.
                     If 'random', uses random uniform noise as baseline.
                     If tensor, uses the provided tensor as baseline.
            *args: Additional positional arguments for the base class
            **kwargs: Additional keyword arguments for the base class
        """
        super().__init__(model, *args, **kwargs)
        self.n_steps = n_steps
        self.baseline = baseline
        
    def _get_baseline(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the baseline input for the path integral.
        
        Args:
            X: Input tensor for which to generate a baseline
            
        Returns:
            Baseline tensor with same shape as X
        """
        if isinstance(self.baseline, torch.Tensor):
            if self.baseline.shape != X.shape:
                raise ValueError(f"Baseline tensor shape {self.baseline.shape} must match input shape {X.shape}")
            return self.baseline
        elif self.baseline == "zero":
            return torch.zeros_like(X)
        elif self.baseline == "random":
            return torch.rand_like(X)
        else:
            raise ValueError(f"Invalid baseline: {self.baseline}. Must be 'zero', 'random', or a tensor")
    
    def generate_maps(
        self, 
        X: torch.Tensor, 
        class_idx: Optional[List[int]] = None, 
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate saliency maps using the Integrated Gradients method.
        
        Args:
            X: Input images to generate attribution maps for
            class_idx: Optional class indices to generate attribution maps for
                      If None, uses the classes predicted by the model
                      If provided but length doesn't match batch size, defaults to predicted classes
            *args: Additional positional arguments for derived classes
            **kwargs: Additional keyword arguments for derived classes
            
        Returns:
            Tuple containing:
                - Attribution maps for the input images
                - Target classes used for generating the maps
        """
        self.model.eval()
        batch_size = X.shape[0]
        
        # Get baseline input
        baseline = self._get_baseline(X)
        
        # Get target classes (predicted or specified)
        with torch.no_grad():
            model_output = self.model(X)
            
            if class_idx is None:
                # Use model's predicted class (highest probability)
                _, predicted_classes = torch.max(model_output, dim=1)
                target_classes = predicted_classes
            else:
                # Check if provided class_idx matches batch size
                if len(class_idx) != batch_size:
                    warnings.warn(
                        f"Length of class_idx ({len(class_idx)}) does not match batch size ({batch_size}). "
                        "Defaulting to predicted classes."
                    )
                    _, predicted_classes = torch.max(model_output, dim=1)
                    target_classes = predicted_classes
                else:
                    target_classes = torch.tensor(class_idx, device=X.device)
        
        # Initialize the integrated gradients
        integrated_grads = torch.zeros_like(X)
        
        # Compute the integral approximation
        for step in range(self.n_steps):
            # Compute the interpolated inputs
            alpha = step / self.n_steps
            interpolated = baseline + alpha * (X - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            
            # Create a one-hot encoding for the target classes
            one_hot = torch.zeros_like(output)
            for i in range(batch_size):
                one_hot[i, target_classes[i]] = 1
            
            # Compute gradients
            self.model.zero_grad()
            output.backward(gradient=one_hot)
            
            # Accumulate gradients
            integrated_grads += interpolated.grad.detach()
            
            # Clean up
            interpolated.requires_grad_(False)
        
        # Scale the integrated gradients
        integrated_grads = integrated_grads * (X - baseline) / self.n_steps
        
        # Normalize to create saliency maps
        saliency_maps = integrated_grads.abs().mean(dim=1)
        
        return saliency_maps, target_classes
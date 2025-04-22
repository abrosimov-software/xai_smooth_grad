import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List, Any
import os
from PIL import Image
import torchvision.transforms as transforms
from src.grads import BaseGrad, SmoothGrad

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor from ImageNet normalization.
    
    Args:
        tensor: Normalized input tensor
    
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return tensor * std + mean

def get_transform() -> transforms.Compose:
    """
    Get the standard ImageNet preprocessing transform.
    
    Returns:
        Composition of transforms for preprocessing images
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def read_images(image_folder: str) -> List[str]:
    """
    Read image paths from a folder.
    
    Args:
        image_folder: Path to the folder containing images
    
    Returns:
        List of image file paths
    """
    return [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]


def create_dataset(
        image_folder: str, 
        device: torch.device, 
        transform: Optional[transforms.Compose] = None
    ) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for image processing.
    
    Args:
        image_folder: Path to the folder containing images
        device: Device to load the DataLoader on
        transform: Preprocessing transformations to apply
    Returns:
        DataLoader for image processing
    """
    img_paths = read_images(image_folder)
    if transform is None:
        transform = get_transform()
    dataset = torch.utils.data.TensorDataset(
        torch.stack([
            transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
            for img_path in img_paths
        ]).to(device)
    )
    return dataset


def visualize_saliency_map(
        original_img: torch.Tensor,
        saliency_map: torch.Tensor,
        pred_class: torch.Tensor,
        figsize: Tuple[int, int] = (10, 5)
    ) -> None:
    """
    Visualize saliency maps.
    
    Args:
        original_img: Original image
        saliency_map: Saliency map
        pred_class: Predicted class
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    denormalized_img = denormalize(original_img).cpu().permute(1, 2, 0).numpy()
    denormalized_img = np.clip(denormalized_img, 0, 1)
    axs[0].imshow(denormalized_img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(saliency_map.cpu().numpy(), cmap="hot")
    axs[1].set_title("Saliency Map")
    plt.suptitle(f"Predicted class: {pred_class.cpu().numpy()}")
    plt.show()


def compare_saliency_methods(
        imgs_dataset: torch.utils.data.Dataset,
        base_methods: List[BaseGrad],
        smooth_methods: List[SmoothGrad],
        figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Compare saliency methods.
    Visualize the saliency maps of the original images for each method in a grid.

    Args:
        imgs_dataset: Dataset of original images
        base_methods: Base saliency methods
        smooth_methods: Smooth saliency methods
        figsize: Figure size
    """
    # Extract images from dataset
    images = [img.squeeze(0) for img,  in imgs_dataset]
    n_images = len(images)
    n_methods = len(base_methods) + len(smooth_methods)
    
    # Create figure with rows = methods + 1 (for original images), cols = number of images
    fig, axs = plt.subplots(n_methods + 1, n_images, figsize=figsize)
    
    # Handle case where there's only one image (axs would be 1D)
    if n_images == 1:
        axs = np.expand_dims(axs, axis=1)
    
    # Display original images in the first row
    for i, img in enumerate(images):
        denormalized_img = denormalize(img).cpu().permute(1, 2, 0).numpy()
        denormalized_img = np.clip(denormalized_img, 0, 1)
        axs[0, i].imshow(denormalized_img)
        axs[0, i].axis("off")
        if i == 0:
            axs[0, i].set_title("Original Images", fontsize=12)
    
    # Create tensor batch for model input
    batch = torch.stack(images)
    
    # Process base methods
    for j, method in enumerate(base_methods):
        saliency_maps, target_classes = method.generate_maps(batch)
        
        for i in range(n_images):
            axs[j+1, i].imshow(saliency_maps[i].cpu().numpy(), cmap="hot")
            axs[j+1, i].axis("off")
            if i == 0:
                axs[j+1, i].set_title(method.__class__.__name__, fontsize=12)
    
    # Process smooth methods
    for j, method in enumerate(smooth_methods):
        saliency_maps, target_classes = method.generate_maps(batch)
        
        for i in range(n_images):
            axs[j+len(base_methods)+1, i].imshow(saliency_maps[i].cpu().numpy(), cmap="hot")
            axs[j+len(base_methods)+1, i].axis("off")
            if i == 0:
                axs[j+len(base_methods)+1, i].set_title(f"Smooth {method.base_grad_class.__class__.__name__}", fontsize=12)
    
    plt.tight_layout()
    plt.show()
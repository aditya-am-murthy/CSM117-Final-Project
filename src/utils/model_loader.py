"""
Model loading utilities for experiments.

This module provides utilities for loading pre-trained models, particularly
Vision Transformers from HuggingFace.
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTFeatureExtractor
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ViTWrapper(nn.Module):
    """Wrapper for HuggingFace ViT model to make it compatible with standard PyTorch usage."""
    
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model
    
    def forward(self, x):
        """Forward pass that returns logits directly."""
        outputs = self.vit_model(x)
        # HuggingFace models return a dictionary-like object with 'logits'
        if hasattr(outputs, 'logits'):
            return outputs.logits
        elif isinstance(outputs, dict):
            return outputs['logits']
        else:
            return outputs
    
    def train(self, mode: bool = True):
        """Set training mode for both wrapper and underlying model."""
        super().train(mode)
        self.vit_model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode for both wrapper and underlying model."""
        super().eval()
        self.vit_model.eval()
        return self


def load_vit_model(model_name: str = "google/vit-base-patch16-224",
                   num_classes: int = 10,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu",
                   replace_classifier: bool = True) -> nn.Module:
    """
    Load a Vision Transformer model from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
            - Use "google/vit-base-patch16-224" for ImageNet-pretrained (recommended for unlearning)
            - Use "nateraw/vit-base-patch16-224-cifar10" for CIFAR-10 fine-tuned (not recommended)
        num_classes: Number of output classes (will replace classifier if replace_classifier=True)
        device: Device to load model on
        replace_classifier: If True, replace the final classifier layer with one for num_classes
        
    Returns:
        Loaded ViT model wrapped for compatibility
    """
    try:
        logger.info(f"Loading ViT model: {model_name}")
        vit_model = ViTForImageClassification.from_pretrained(model_name)
        
        # Replace classifier head if needed (e.g., ImageNet has 1000 classes, CIFAR-10 has 10)
        if replace_classifier and hasattr(vit_model, 'classifier'):
            original_in_features = vit_model.classifier.in_features
            vit_model.classifier = nn.Linear(original_in_features, num_classes)
            logger.info(f"Replaced classifier head: {original_in_features} -> {num_classes} classes")
        
        vit_model.to(device)
        vit_model.eval()
        
        # Wrap the model for compatibility
        wrapped_model = ViTWrapper(vit_model)
        wrapped_model.to(device)
        
        logger.info(f"Successfully loaded ViT model on {device}")
        return wrapped_model
    except Exception as e:
        logger.error(f"Failed to load ViT model: {e}")
        raise


def get_vit_feature_extractor(model_name: str = "nateraw/vit-base-patch16-224-cifar10"):
    """
    Get ViT feature extractor for preprocessing images.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        ViT feature extractor
    """
    try:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        return feature_extractor
    except Exception as e:
        logger.error(f"Failed to load ViT feature extractor: {e}")
        raise


def create_model_from_config(model_name: str, num_classes: int = 10,
                            device: str = "cuda" if torch.cuda.is_available() else "cpu",
                            replace_classifier: bool = True) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        model_name: Model name or identifier
            - "google/vit-base-patch16-224": ImageNet-pretrained ViT (recommended)
            - "nateraw/vit-base-patch16-224-cifar10": CIFAR-10 fine-tuned (not recommended for unlearning)
        num_classes: Number of output classes
        device: Device to create model on
        replace_classifier: Whether to replace classifier head (for ViT models)
        
    Returns:
        Model instance
    """
    if "vit" in model_name.lower() or "google" in model_name.lower() or "nateraw" in model_name.lower():
        # For unlearning experiments, we want ImageNet-pretrained, NOT CIFAR-10 fine-tuned
        if "cifar10" in model_name.lower():
            logger.warning(
                f"Warning: Using model '{model_name}' which was already fine-tuned on CIFAR-10. "
                "For proper unlearning evaluation, consider using 'google/vit-base-patch16-224' "
                "which is pretrained on ImageNet but not fine-tuned on CIFAR-10."
            )
        return load_vit_model(model_name, num_classes, device, replace_classifier=replace_classifier)
    elif model_name.lower() == "resnet18":
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(device)
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")


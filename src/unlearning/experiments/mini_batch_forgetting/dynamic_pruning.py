"""
Dynamic Pruning for Mini-batch Forgetting.

This module implements adaptive pruning techniques that identify and isolate
forget regions in the model by dynamically pruning neurons/parameters that
are most influenced by the forget data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import copy
import numpy as np
from collections import defaultdict

from ....fl.base import UnlearningStrategy, FLConfig


class DynamicPruningUnlearning(UnlearningStrategy):
    """
    Dynamic pruning unlearning strategy that adaptively isolates forget regions.
    
    This method:
    1. Computes importance scores for each parameter based on forget data
    2. Dynamically prunes parameters with high forget importance
    3. Fine-tunes on retain data to maintain performance
    """
    
    def __init__(self, config: FLConfig, 
                 pruning_ratio: float = 0.1,
                 importance_threshold: float = 0.5,
                 fine_tune_epochs: int = 5):
        super().__init__(config)
        self.pruning_ratio = pruning_ratio
        self.importance_threshold = importance_threshold
        self.fine_tune_epochs = fine_tune_epochs
        self.pruning_mask: Dict[str, torch.Tensor] = {}
    
    def _compute_parameter_importance(self, model: nn.Module, 
                                     forget_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for each parameter based on forget data.
        
        Uses gradient-based importance: parameters with large gradients on forget data
        are considered important for forgetting.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        model.eval()
        importance_scores = {}
        
        # Initialize importance scores
        for name, param in model.named_parameters():
            if param.requires_grad:
                importance_scores[name] = torch.zeros_like(param)
        
        criterion = nn.CrossEntropyLoss()
        num_batches = 0
        total_batches = len(forget_loader)
        
        for batch_idx, (data, target) in enumerate(forget_loader):
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Accumulate gradient magnitudes as importance
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    importance_scores[name] += torch.abs(param.grad)
            
            num_batches += 1
            
            # Log progress every 10% of batches
            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                logger.info(f"    Processed {batch_idx+1}/{total_batches} batches ({100*(batch_idx+1)/total_batches:.1f}%)")
        
        # Normalize by number of batches
        for name in importance_scores:
            importance_scores[name] /= max(num_batches, 1)
        
        return importance_scores
    
    def _create_pruning_mask(self, importance_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Create pruning mask based on importance scores.
        
        Parameters with importance above threshold are marked for pruning.
        """
        pruning_mask = {}
        
        for name, importance in importance_scores.items():
            # Normalize importance scores
            normalized_importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            
            # Create mask: 1 means keep, 0 means prune
            mask = (normalized_importance < self.importance_threshold).float()
            
            # Ensure at least some parameters are kept
            num_params = mask.numel()
            num_to_keep = max(1, int(num_params * (1 - self.pruning_ratio)))
            
            if mask.sum() < num_to_keep:
                # Keep top-k least important parameters
                flat_importance = normalized_importance.flatten()
                _, indices = torch.topk(flat_importance, num_to_keep, largest=False)
                mask = torch.zeros_like(normalized_importance).flatten()
                mask[indices] = 1.0
                mask = mask.reshape(normalized_importance.shape)
            
            pruning_mask[name] = mask
        
        return pruning_mask
    
    def _apply_pruning(self, model: nn.Module, pruning_mask: Dict[str, torch.Tensor]):
        """Apply pruning mask to model parameters."""
        for name, param in model.named_parameters():
            if name in pruning_mask:
                mask = pruning_mask[name].to(param.device)
                param.data *= mask
    
    def _fine_tune_on_retain(self, model: nn.Module, retain_loader: DataLoader):
        """Fine-tune model on retain data while maintaining pruning mask."""
        import logging
        logger = logging.getLogger(__name__)
        
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.1)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.fine_tune_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(retain_loader):
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply gradient masking to respect pruning
                for name, param in model.named_parameters():
                    if name in self.pruning_mask:
                        mask = self.pruning_mask[name].to(param.device)
                        if param.grad is not None:
                            param.grad *= mask
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Log progress every 10% of batches
                if (batch_idx + 1) % max(1, len(retain_loader) // 10) == 0:
                    logger.info(f"  Epoch {epoch+1}/{self.fine_tune_epochs}, Batch {batch_idx+1}/{len(retain_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"  Epoch {epoch+1}/{self.fine_tune_epochs} complete, Average Loss: {avg_loss:.4f}")
    
    def unlearn(self, model: nn.Module, forget_data: DataLoader,
                retain_data: DataLoader) -> nn.Module:
        """
        Perform dynamic pruning unlearning.
        
        Steps:
        1. Compute parameter importance on forget data
        2. Create pruning mask
        3. Apply pruning
        4. Fine-tune on retain data
        """
        unlearned_model = copy.deepcopy(model)
        unlearned_model.to(self.config.device)
        
        # Step 1: Compute importance scores
        importance_scores = self._compute_parameter_importance(unlearned_model, forget_data)
        
        # Step 2: Create pruning mask
        self.pruning_mask = self._create_pruning_mask(importance_scores)
        
        # Step 3: Apply pruning
        self._apply_pruning(unlearned_model, self.pruning_mask)
        
        # Step 4: Fine-tune on retain data
        self._fine_tune_on_retain(unlearned_model, retain_data)
        
        return unlearned_model
    
    def evaluate_unlearning(self, model: nn.Module, forget_data: DataLoader,
                           retain_data: DataLoader) -> Dict[str, float]:
        """Evaluate dynamic pruning unlearning effectiveness."""
        model.eval()
        device = self.config.device
        
        # Evaluate on forget data
        forget_correct = 0
        forget_total = 0
        forget_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in forget_data:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                forget_loss += loss.item()
                
                pred = output.argmax(dim=1)
                forget_correct += (pred == target).sum().item()
                forget_total += target.size(0)
        
        # Evaluate on retain data
        retain_correct = 0
        retain_total = 0
        retain_loss = 0.0
        
        with torch.no_grad():
            for data, target in retain_data:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                retain_loss += loss.item()
                
                pred = output.argmax(dim=1)
                retain_correct += (pred == target).sum().item()
                retain_total += target.size(0)
        
        forget_accuracy = forget_correct / max(forget_total, 1)
        retain_accuracy = retain_correct / max(retain_total, 1)
        
        return {
            'forget_accuracy': forget_accuracy,
            'retain_accuracy': retain_accuracy,
            'forget_loss': forget_loss / len(forget_data) if len(forget_data) > 0 else 0.0,
            'retain_loss': retain_loss / len(retain_data) if len(retain_data) > 0 else 0.0,
            'unlearning_effectiveness': 1.0 - forget_accuracy,
            'retention_preservation': retain_accuracy,
            'pruning_ratio': self.pruning_ratio,
            'num_pruned_params': sum(mask.sum().item() for mask in self.pruning_mask.values())
        }


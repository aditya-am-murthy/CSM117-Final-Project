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
                 fine_tune_epochs: int = 5,
                 forget_loss_weight: float = 1.0,
                 prune_classifier_only: bool = False):
        super().__init__(config)
        self.pruning_ratio = pruning_ratio
        self.importance_threshold = importance_threshold
        self.fine_tune_epochs = fine_tune_epochs
        self.forget_loss_weight = forget_loss_weight  # Weight for forgetting penalty during fine-tuning
        self.prune_classifier_only = prune_classifier_only  # If True, only prune classifier head
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
        
        Prunes parameters with HIGH importance (most influenced by forget data).
        Mask: 1 means keep, 0 means prune.
        
        If prune_classifier_only is True, only prunes classifier/head layers.
        Otherwise, prunes all layers but protects embedding layers.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        pruning_mask = {}
        
        # Layers to protect (never prune) - these are critical for model structure
        protected_keywords = ['embedding', 'pos_embed', 'patch_embed', 'norm', 'layer_norm']
        
        for name, importance in importance_scores.items():
            # Skip if we're only pruning classifier and this isn't a classifier layer
            if self.prune_classifier_only:
                if 'classifier' not in name.lower() and 'head' not in name.lower():
                    # Keep all non-classifier parameters
                    pruning_mask[name] = torch.ones_like(importance)
                    continue
            
            # Protect critical layers (embeddings, normalization)
            should_protect = any(keyword in name.lower() for keyword in protected_keywords)
            if should_protect:
                logger.debug(f"  Protecting layer: {name}")
                pruning_mask[name] = torch.ones_like(importance)
                continue
            
            # Normalize importance scores to [0, 1]
            if importance.max() > importance.min():
                normalized_importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            else:
                normalized_importance = torch.zeros_like(importance)
            
            # Calculate how many parameters to keep (prune the rest)
            num_params = importance.numel()
            num_to_keep = max(1, int(num_params * (1 - self.pruning_ratio)))
            
            # Prune parameters with HIGHEST importance (most important for forget data)
            # Keep parameters with LOWEST importance
            flat_importance = normalized_importance.flatten()
            _, indices = torch.topk(flat_importance, num_to_keep, largest=False)  # Get indices of least important
            
            # Create mask: 1 for keep (low importance), 0 for prune (high importance)
            mask = torch.zeros_like(normalized_importance).flatten()
            mask[indices] = 1.0
            mask = mask.reshape(normalized_importance.shape)
            
            pruning_mask[name] = mask
            
            # Log pruning statistics
            pruned_count = (mask == 0).sum().item()
            total_count = mask.numel()
            logger.debug(f"  Layer {name}: Pruned {pruned_count}/{total_count} params ({100*pruned_count/total_count:.1f}%)")
        
        return pruning_mask
    
    def _apply_pruning(self, model: nn.Module, pruning_mask: Dict[str, torch.Tensor]):
        """Apply pruning mask to model parameters."""
        for name, param in model.named_parameters():
            if name in pruning_mask:
                mask = pruning_mask[name].to(param.device)
                param.data *= mask
    
    def _fine_tune_on_retain(self, model: nn.Module, retain_loader: DataLoader, forget_loader: DataLoader):
        """
        Fine-tune model on retain data while maintaining pruning mask.
        Also applies forgetting penalty to reduce accuracy on forget data.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        model.train()
        # Use a smaller learning rate for fine-tuning to be more gentle
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.05)
        criterion = nn.CrossEntropyLoss()
        
        # Create iterator for forget data (cycle through it)
        forget_iter = iter(forget_loader)
        
        # Evaluate initial retention accuracy before fine-tuning
        model.eval()
        initial_retain_correct = 0
        initial_retain_total = 0
        with torch.no_grad():
            for data, target in retain_loader:
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                output = model(data)
                pred = output.argmax(dim=1)
                initial_retain_correct += (pred == target).sum().item()
                initial_retain_total += target.size(0)
        best_retain_acc = initial_retain_correct / max(initial_retain_total, 1)
        logger.info(f"  Initial retention accuracy before fine-tuning: {best_retain_acc:.4f}")
        model.train()
        
        for epoch in range(self.fine_tune_epochs):
            epoch_loss = 0.0
            epoch_retain_loss = 0.0
            epoch_forget_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(retain_loader):
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                
                optimizer.zero_grad()
                
                # Retain loss: maximize accuracy on retain data
                output = model(data)
                retain_loss = criterion(output, target)
                
                # Forgetting loss: minimize accuracy on forget data (penalize correct predictions)
                # This explicitly encourages the model to forget
                try:
                    forget_data, forget_target = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_data, forget_target = next(forget_iter)
                
                forget_data = forget_data.to(self.config.device)
                forget_target = forget_target.to(self.config.device)
                forget_output = model(forget_data)
                
                # Gentle forgetting loss: minimize the probability assigned to the correct class
                # This encourages the model to be less confident on forget data
                # without completely destroying its ability to predict
                forget_probs = torch.softmax(forget_output, dim=1)
                correct_class_probs = forget_probs.gather(1, forget_target.unsqueeze(1)).squeeze(1)
                
                # Forgetting loss: minimize correct class probability
                # This is gentler than maximizing CE loss - it just reduces confidence
                forget_loss = correct_class_probs.mean()
                
                # Combined loss: prioritize retention, gently reduce forget accuracy
                # The forget_loss_weight should be small to avoid breaking the model
                total_loss = retain_loss + self.forget_loss_weight * forget_loss
                total_loss.backward()
                
                # Apply gradient masking to respect pruning (don't update pruned parameters)
                for name, param in model.named_parameters():
                    if name in self.pruning_mask:
                        mask = self.pruning_mask[name].to(param.device)
                        if param.grad is not None:
                            param.grad *= mask
                
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_retain_loss += retain_loss.item()
                epoch_forget_loss += forget_loss.item()
                num_batches += 1
                
                # Log progress every 10% of batches
                if (batch_idx + 1) % max(1, len(retain_loader) // 10) == 0:
                    logger.info(f"  Epoch {epoch+1}/{self.fine_tune_epochs}, Batch {batch_idx+1}/{len(retain_loader)}, "
                               f"Total Loss: {total_loss.item():.4f}, Retain: {retain_loss.item():.4f}, Forget: {forget_loss.item():.4f}")
            
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_retain_loss = epoch_retain_loss / max(num_batches, 1)
            avg_forget_loss = epoch_forget_loss / max(num_batches, 1)
            
            # Evaluate retention accuracy periodically to monitor performance
            if epoch % 2 == 0 or epoch == self.fine_tune_epochs - 1:
                model.eval()
                retain_correct = 0
                retain_total = 0
                with torch.no_grad():
                    for data, target in retain_loader:
                        data = data.to(self.config.device)
                        target = target.to(self.config.device)
                        output = model(data)
                        pred = output.argmax(dim=1)
                        retain_correct += (pred == target).sum().item()
                        retain_total += target.size(0)
                retain_acc = retain_correct / max(retain_total, 1)
                model.train()
                
                logger.info(f"  Epoch {epoch+1}/{self.fine_tune_epochs} complete - "
                           f"Avg Total Loss: {avg_loss:.4f}, Retain Loss: {avg_retain_loss:.4f}, "
                           f"Forget Loss: {avg_forget_loss:.4f}, Retain Acc: {retain_acc:.4f}")
                
                # Early stopping if retention drops too much
                if retain_acc < best_retain_acc - 0.05:  # 5% drop threshold
                    logger.warning(f"  Retention accuracy dropped significantly ({retain_acc:.4f} < {best_retain_acc:.4f} - 0.05)")
                    logger.warning(f"  Stopping fine-tuning early to preserve model performance")
                    break
                
                if retain_acc > best_retain_acc:
                    best_retain_acc = retain_acc
            else:
                logger.info(f"  Epoch {epoch+1}/{self.fine_tune_epochs} complete - "
                           f"Avg Total Loss: {avg_loss:.4f}, Retain Loss: {avg_retain_loss:.4f}, Forget Loss: {avg_forget_loss:.4f}")
    
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
        
        # Step 4: Fine-tune on retain data with forgetting penalty
        self._fine_tune_on_retain(unlearned_model, retain_data, forget_data)
        
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


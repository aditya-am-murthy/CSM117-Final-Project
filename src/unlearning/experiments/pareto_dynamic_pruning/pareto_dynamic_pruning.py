"""
Pareto Optimization with Dynamic Pruning for Multi-objective Unlearning.

This module combines dynamic pruning with Pareto optimization:
1. First applies dynamic pruning to identify and isolate forget regions
2. Then uses Pareto optimization to fine-tune the pruned model, optimizing
   the trade-off between forgetting and retention objectives.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import copy
import numpy as np
from collections import defaultdict
import logging

from ....fl.base import UnlearningStrategy, FLConfig
from ..mini_batch_forgetting.dynamic_pruning import DynamicPruningUnlearning


class ParetoDynamicPruningUnlearning(UnlearningStrategy):
    """
    Combined dynamic pruning + Pareto optimization unlearning strategy.
    
    This method:
    1. Uses dynamic pruning to identify and prune parameters important for forget data
    2. Applies Pareto optimization to fine-tune the pruned model, finding optimal
       trade-offs between forgetting and retention objectives
    """
    
    def __init__(self, config: FLConfig,
                 # Dynamic pruning parameters
                 pruning_ratio: float = 0.1,
                 importance_threshold: float = 0.5,
                 prune_classifier_only: bool = False,
                 # Pareto optimization parameters
                 forget_weight: float = 0.5,
                 retention_weight: float = 0.5,
                 pareto_steps: int = 20,
                 adaptive_weights: bool = True,
                 unlearning_epochs: int = 10):
        super().__init__(config)
        
        # Dynamic pruning parameters
        self.pruning_ratio = pruning_ratio
        self.importance_threshold = importance_threshold
        self.prune_classifier_only = prune_classifier_only
        
        # Pareto optimization parameters
        self.forget_weight = forget_weight
        self.retention_weight = retention_weight
        self.pareto_steps = pareto_steps
        self.adaptive_weights = adaptive_weights
        self.unlearning_epochs = unlearning_epochs
        
        # Internal state
        self.pareto_frontier: List[Dict[str, Any]] = []
        self.current_weights: Tuple[float, float] = (forget_weight, retention_weight)
        self.pruning_mask: Dict[str, torch.Tensor] = {}
        
        # Create dynamic pruning strategy for the pruning phase
        self.dynamic_pruning = DynamicPruningUnlearning(
            config=config,
            pruning_ratio=pruning_ratio,
            importance_threshold=importance_threshold,
            fine_tune_epochs=0,  # We'll do fine-tuning with Pareto optimization instead
            forget_loss_weight=0.0,  # No forgetting penalty during pruning phase
            prune_classifier_only=prune_classifier_only
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _compute_multi_objective_loss(self, model: nn.Module, forget_loader: DataLoader,
                                     retain_loader: DataLoader) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-objective loss combining forgetting and retention objectives.
        
        Returns:
            Total loss and individual objective values
        """
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        # Compute forget objective (we want to maximize loss, so minimize negative)
        forget_losses = []
        for data, target in forget_loader:
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            output = model(data)
            loss = criterion(output, target)
            forget_losses.append(loss)
        
        if forget_losses:
            forget_obj = torch.stack(forget_losses).mean()
        else:
            forget_obj = torch.tensor(0.0, device=self.config.device)
        
        # Compute retention objective (we want to minimize loss)
        retention_losses = []
        for data, target in retain_loader:
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            output = model(data)
            loss = criterion(output, target)
            retention_losses.append(loss)
        
        if retention_losses:
            retention_obj = torch.stack(retention_losses).mean()
        else:
            retention_obj = torch.tensor(0.0, device=self.config.device)
        
        # Multi-objective: maximize forgetting (minimize negative) + minimize retention
        total_loss = -self.current_weights[0] * forget_obj + self.current_weights[1] * retention_obj
        
        objectives = {
            'forget_loss': forget_obj.item(),
            'retention_loss': retention_obj.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, objectives
    
    def _update_adaptive_weights(self, forget_loss: float, retention_loss: float,
                                 forget_accuracy: float, retain_accuracy: float):
        """
        Adaptively update weights based on current performance.
        
        If forgetting is poor, increase forget weight.
        If retention is poor, increase retention weight.
        """
        if not self.adaptive_weights:
            return
        
        # Normalize losses to [0, 1] range (approximate)
        forget_norm = min(forget_loss / 5.0, 1.0)  # Assuming max loss ~5.0
        retention_norm = min(retention_loss / 5.0, 1.0)
        
        # Adjust weights: if forget accuracy is high (low loss), we can focus on retention
        # If forget accuracy is low (high loss), focus on forgetting
        forget_weight = 0.3 + 0.4 * forget_norm  # Range: [0.3, 0.7]
        retention_weight = 0.3 + 0.4 * retention_norm  # Range: [0.3, 0.7]
        
        # Normalize to sum to 1.0
        total = forget_weight + retention_weight
        self.current_weights = (forget_weight / total, retention_weight / total)
    
    def _evaluate_accuracy(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Evaluate model accuracy on a data loader."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return correct / max(total, 1)
    
    def _pareto_optimization_step(self, model: nn.Module, forget_loader: DataLoader,
                                 retain_loader: DataLoader):
        """
        Perform Pareto optimization on the pruned model.
        
        Uses gradient descent on the multi-objective loss while respecting the pruning mask.
        """
        # Use a smaller learning rate for fine-tuning
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.05)
        
        for epoch in range(self.unlearning_epochs):
            # Compute multi-objective loss
            total_loss, objectives = self._compute_multi_objective_loss(
                model, forget_loader, retain_loader
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Apply gradient masking to respect pruning (don't update pruned parameters)
            for name, param in model.named_parameters():
                if name in self.pruning_mask:
                    mask = self.pruning_mask[name].to(param.device)
                    if param.grad is not None:
                        param.grad *= mask
            
            optimizer.step()
            
            # Evaluate current performance
            forget_acc = self._evaluate_accuracy(model, forget_loader)
            retain_acc = self._evaluate_accuracy(model, retain_loader)
            
            # Update adaptive weights
            self._update_adaptive_weights(
                objectives['forget_loss'],
                objectives['retention_loss'],
                forget_acc,
                retain_acc
            )
            
            # Store Pareto point
            pareto_point = {
                'forget_loss': objectives['forget_loss'],
                'retention_loss': objectives['retention_loss'],
                'forget_accuracy': forget_acc,
                'retain_accuracy': retain_acc,
                'forget_weight': self.current_weights[0],
                'retention_weight': self.current_weights[1],
                'epoch': epoch
            }
            self.pareto_frontier.append(pareto_point)
            
            # Log progress
            if (epoch + 1) % max(1, self.unlearning_epochs // 5) == 0 or epoch == 0:
                self.logger.info(f"  Pareto Epoch {epoch+1}/{self.unlearning_epochs}: "
                               f"Forget Loss: {objectives['forget_loss']:.4f}, "
                               f"Retention Loss: {objectives['retention_loss']:.4f}, "
                               f"Forget Acc: {forget_acc:.4f}, Retain Acc: {retain_acc:.4f}")
    
    def unlearn(self, model: nn.Module, forget_data: DataLoader,
                retain_data: DataLoader) -> nn.Module:
        """
        Perform combined dynamic pruning + Pareto optimization unlearning.
        
        Steps:
        1. Apply dynamic pruning to identify and prune forget-important parameters
        2. Use Pareto optimization to fine-tune the pruned model
        """
        unlearned_model = copy.deepcopy(model)
        unlearned_model.to(self.config.device)
        
        # Step 1: Apply dynamic pruning
        self.logger.info("Step 1/2: Applying dynamic pruning...")
        self.logger.info(f"  Computing parameter importance on forget data...")
        
        # Get the pruning mask from dynamic pruning
        importance_scores = self.dynamic_pruning._compute_parameter_importance(unlearned_model, forget_data)
        self.pruning_mask = self.dynamic_pruning._create_pruning_mask(importance_scores)
        
        # Apply pruning
        self.dynamic_pruning._apply_pruning(unlearned_model, self.pruning_mask)
        self.logger.info("  Dynamic pruning complete")
        
        # Step 2: Apply Pareto optimization
        self.logger.info(f"Step 2/2: Applying Pareto optimization ({self.unlearning_epochs} epochs)...")
        self.logger.info(f"  Forget batches: {len(forget_data)}, Retain batches: {len(retain_data)}")
        self._pareto_optimization_step(unlearned_model, forget_data, retain_data)
        self.logger.info("  Pareto optimization complete")
        
        self.logger.info("Combined dynamic pruning + Pareto optimization unlearning completed successfully")
        return unlearned_model
    
    def evaluate_unlearning(self, model: nn.Module, forget_data: DataLoader,
                           retain_data: DataLoader) -> Dict[str, float]:
        """
        Evaluate unlearning performance.
        
        Returns metrics including Pareto frontier information.
        """
        forget_acc = self._evaluate_accuracy(model, forget_data)
        retain_acc = self._evaluate_accuracy(model, retain_data)
        
        metrics = {
            'forget_accuracy': forget_acc,
            'retain_accuracy': retain_acc,
            'pareto_frontier_size': len(self.pareto_frontier)
        }
        
        if self.pareto_frontier:
            # Get the final Pareto point
            final_point = self.pareto_frontier[-1]
            metrics.update({
                'final_forget_loss': final_point['forget_loss'],
                'final_retention_loss': final_point['retention_loss'],
                'final_forget_weight': final_point['forget_weight'],
                'final_retention_weight': final_point['retention_weight']
            })
        
        return metrics


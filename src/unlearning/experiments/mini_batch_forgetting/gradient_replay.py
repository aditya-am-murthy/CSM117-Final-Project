"""
Gradient Replay Buffer for Mini-batch Forgetting.

This module implements an adaptive gradient replay buffer that isolates forget
regions by maintaining a buffer of gradients from retain data and using them
to counteract forget gradients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Deque
import copy
import numpy as np
from collections import deque

from ....fl.base import UnlearningStrategy, FLConfig


class GradientReplayBufferUnlearning(UnlearningStrategy):
    """
    Gradient replay buffer unlearning strategy.
    
    This method:
    1. Maintains a buffer of gradients from retain data
    2. Computes gradients on forget data
    3. Uses replay gradients to counteract forget gradients adaptively
    4. Isolates forget regions through gradient interference
    """
    
    def __init__(self, config: FLConfig,
                 buffer_size: int = 100,
                 replay_weight: float = 0.5,
                 adaptive_threshold: float = 0.1,
                 unlearning_epochs: int = 10):
        super().__init__(config)
        self.buffer_size = buffer_size
        self.replay_weight = replay_weight
        self.adaptive_threshold = adaptive_threshold
        self.unlearning_epochs = unlearning_epochs
        self.gradient_buffer: Deque[Dict[str, torch.Tensor]] = deque(maxlen=buffer_size)
        self.forget_region_mask: Dict[str, torch.Tensor] = {}
    
    def _compute_gradients(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Compute gradients for a batch of data."""
        model.train()
        gradients = {}
        
        # Initialize gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                gradients[name] = torch.zeros_like(param)
        
        criterion = nn.CrossEntropyLoss()
        num_samples = 0
        
        for data, target in data_loader:
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Accumulate gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradients[name] += param.grad * data.size(0)
            
            num_samples += data.size(0)
        
        # Normalize by number of samples
        for name in gradients:
            gradients[name] /= max(num_samples, 1)
        
        return gradients
    
    def _update_gradient_buffer(self, model: nn.Module, retain_loader: DataLoader):
        """Update gradient replay buffer with gradients from retain data."""
        # Compute gradients on retain data and add to buffer
        gradients = self._compute_gradients(model, retain_loader)
        self.gradient_buffer.append(gradients)
    
    def _identify_forget_regions(self, forget_gradients: Dict[str, torch.Tensor],
                                retain_gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Identify forget regions by comparing forget and retain gradients.
        
        Regions where forget gradients are significantly different from retain gradients
        are marked as forget regions.
        """
        forget_region_mask = {}
        
        for name in forget_gradients:
            if name not in retain_gradients:
                continue
            
            forget_grad = forget_gradients[name]
            retain_grad = retain_gradients[name]
            
            # Compute gradient difference
            grad_diff = torch.abs(forget_grad - retain_grad)
            
            # Normalize
            max_diff = grad_diff.max()
            if max_diff > 0:
                normalized_diff = grad_diff / max_diff
            else:
                normalized_diff = grad_diff
            
            # Mark regions where difference exceeds threshold
            mask = (normalized_diff > self.adaptive_threshold).float()
            forget_region_mask[name] = mask
        
        return forget_region_mask
    
    def _adaptive_gradient_update(self, model: nn.Module, forget_loader: DataLoader,
                                 retain_loader: DataLoader):
        """
        Perform adaptive gradient update using replay buffer.
        
        Combines forget gradients (negated) with replay gradients to isolate forget regions.
        """
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Compute average retain gradients from buffer
        if self.gradient_buffer:
            avg_retain_grad = {}
            for name in self.gradient_buffer[0]:
                avg_retain_grad[name] = torch.zeros_like(self.gradient_buffer[0][name])
                for grad_dict in self.gradient_buffer:
                    avg_retain_grad[name] += grad_dict[name]
                avg_retain_grad[name] /= len(self.gradient_buffer)
        else:
            # If buffer is empty, compute from retain loader
            avg_retain_grad = self._compute_gradients(model, retain_loader)
        
        for epoch in range(self.unlearning_epochs):
            # Compute forget gradients
            forget_gradients = self._compute_gradients(model, forget_loader)
            
            # Identify forget regions
            self.forget_region_mask = self._identify_forget_regions(
                forget_gradients, avg_retain_grad
            )
            
            # Perform gradient update
            for batch_data, batch_target in retain_loader:
                batch_data = batch_data.to(self.config.device)
                batch_target = batch_target.to(self.config.device)
                
                optimizer.zero_grad()
                output = model(batch_data)
                loss = criterion(output, batch_target)
                loss.backward()
                
                # Modify gradients: reduce forget region gradients, enhance retain gradients
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if name in self.forget_region_mask:
                            mask = self.forget_region_mask[name].to(param.device)
                            # Reduce gradients in forget regions
                            param.grad = param.grad * (1 - mask * self.replay_weight)
                            
                            # Add replay gradient in forget regions
                            if name in avg_retain_grad:
                                replay_grad = avg_retain_grad[name].to(param.device)
                                param.grad += replay_grad * mask * self.replay_weight
                
                optimizer.step()
    
    def unlearn(self, model: nn.Module, forget_data: DataLoader,
                retain_data: DataLoader) -> nn.Module:
        """
        Perform gradient replay buffer unlearning.
        
        Steps:
        1. Initialize gradient buffer with retain gradients
        2. Perform adaptive gradient updates
        """
        unlearned_model = copy.deepcopy(model)
        unlearned_model.to(self.config.device)
        
        # Step 1: Initialize gradient buffer
        self._update_gradient_buffer(unlearned_model, retain_data)
        
        # Step 2: Perform adaptive gradient updates
        self._adaptive_gradient_update(unlearned_model, forget_data, retain_data)
        
        return unlearned_model
    
    def evaluate_unlearning(self, model: nn.Module, forget_data: DataLoader,
                           retain_data: DataLoader) -> Dict[str, float]:
        """Evaluate gradient replay buffer unlearning effectiveness."""
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
        
        # Compute forget region statistics
        forget_region_ratio = 0.0
        if self.forget_region_mask:
            total_params = sum(mask.numel() for mask in self.forget_region_mask.values())
            forget_params = sum(mask.sum().item() for mask in self.forget_region_mask.values())
            forget_region_ratio = forget_params / max(total_params, 1)
        
        return {
            'forget_accuracy': forget_accuracy,
            'retain_accuracy': retain_accuracy,
            'forget_loss': forget_loss / len(forget_data) if len(forget_data) > 0 else 0.0,
            'retain_loss': retain_loss / len(retain_data) if len(retain_data) > 0 else 0.0,
            'unlearning_effectiveness': 1.0 - forget_accuracy,
            'retention_preservation': retain_accuracy,
            'forget_region_ratio': forget_region_ratio,
            'buffer_size': len(self.gradient_buffer)
        }


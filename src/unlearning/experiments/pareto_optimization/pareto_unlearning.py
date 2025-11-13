"""
Pareto Optimization for Multi-objective Unlearning.

This module implements Pareto optimization frameworks that jointly optimize
forgetting accuracy and retention fidelity, finding optimal trade-offs between
these objectives.
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


class ParetoOptimizationUnlearning(UnlearningStrategy):
    """
    Pareto optimization unlearning strategy.
    
    This method:
    1. Defines multi-objective loss function (forgetting + retention)
    2. Uses Pareto optimization to find optimal trade-offs
    3. Applies gradient-based optimization with adaptive weights
    4. Tracks Pareto frontier of solutions
    """
    
    def __init__(self, config: FLConfig,
                 forget_weight: float = 0.5,
                 retention_weight: float = 0.5,
                 pareto_steps: int = 20,
                 adaptive_weights: bool = True,
                 unlearning_epochs: int = 10):
        super().__init__(config)
        self.forget_weight = forget_weight
        self.retention_weight = retention_weight
        self.pareto_steps = pareto_steps
        self.adaptive_weights = adaptive_weights
        self.unlearning_epochs = unlearning_epochs
        self.pareto_frontier: List[Dict[str, Any]] = []
        self.current_weights: Tuple[float, float] = (forget_weight, retention_weight)
    
    def _compute_forget_loss(self, model: nn.Module, forget_loader: DataLoader) -> torch.Tensor:
        """Compute loss on forget data (we want to maximize this, i.e., minimize negative)."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in forget_loader:
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                num_batches += 1
        
        return torch.tensor(total_loss / max(num_batches, 1))
    
    def _compute_retention_loss(self, model: nn.Module, retain_loader: DataLoader) -> torch.Tensor:
        """Compute loss on retain data (we want to minimize this)."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in retain_loader:
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                num_batches += 1
        
        return torch.tensor(total_loss / max(num_batches, 1))
    
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
        # We use negative forget loss because we want to maximize it
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
    
    def _pareto_optimization_step(self, model: nn.Module, forget_loader: DataLoader,
                                 retain_loader: DataLoader):
        """
        Perform one step of Pareto optimization.
        
        Uses gradient descent on the multi-objective loss.
        """
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        for epoch in range(self.unlearning_epochs):
            # Compute multi-objective loss
            total_loss, objectives = self._compute_multi_objective_loss(
                model, forget_loader, retain_loader
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
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
    
    def _find_pareto_optimal_solution(self) -> Dict[str, Any]:
        """
        Find the Pareto optimal solution from the frontier.
        
        Selects the solution that best balances forgetting and retention.
        """
        if not self.pareto_frontier:
            return {}
        
        # Find solution with best trade-off (minimize weighted sum)
        best_idx = 0
        best_score = float('inf')
        
        for i, point in enumerate(self.pareto_frontier):
            # Score: lower forget accuracy (better forgetting) + higher retain accuracy (better retention)
            # We want to minimize: (1 - forget_accuracy) + (1 - retain_accuracy)
            score = (1 - point['forget_accuracy']) + (1 - point['retain_accuracy'])
            if score < best_score:
                best_score = score
                best_idx = i
        
        return self.pareto_frontier[best_idx]
    
    def unlearn(self, model: nn.Module, forget_data: DataLoader,
                retain_data: DataLoader) -> nn.Module:
        """
        Perform Pareto optimization unlearning.
        
        Steps:
        1. Initialize weights
        2. Perform Pareto optimization steps
        3. Return optimized model
        """
        unlearned_model = copy.deepcopy(model)
        unlearned_model.to(self.config.device)
        
        # Reset Pareto frontier
        self.pareto_frontier = []
        
        # Initialize weights
        self.current_weights = (self.forget_weight, self.retention_weight)
        
        # Perform Pareto optimization
        self._pareto_optimization_step(unlearned_model, forget_data, retain_data)
        
        return unlearned_model
    
    def evaluate_unlearning(self, model: nn.Module, forget_data: DataLoader,
                           retain_data: DataLoader) -> Dict[str, float]:
        """Evaluate Pareto optimization unlearning effectiveness."""
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
        
        # Get Pareto optimal solution metrics
        pareto_optimal = self._find_pareto_optimal_solution()
        
        return {
            'forget_accuracy': forget_accuracy,
            'retain_accuracy': retain_accuracy,
            'forget_loss': forget_loss / len(forget_data) if len(forget_data) > 0 else 0.0,
            'retain_loss': retain_loss / len(retain_data) if len(retain_data) > 0 else 0.0,
            'unlearning_effectiveness': 1.0 - forget_accuracy,
            'retention_preservation': retain_accuracy,
            'pareto_frontier_size': len(self.pareto_frontier),
            'pareto_optimal_forget_acc': pareto_optimal.get('forget_accuracy', forget_accuracy),
            'pareto_optimal_retain_acc': pareto_optimal.get('retain_accuracy', retain_accuracy),
            'final_forget_weight': self.current_weights[0],
            'final_retention_weight': self.current_weights[1]
        }


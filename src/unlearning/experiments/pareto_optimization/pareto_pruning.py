"""
Hybrid Pareto-Pruning Unlearning Strategy.

This module combines dynamic pruning and Pareto optimization to achieve
optimal unlearning. It first identifies and prunes forget-sensitive parameters,
then applies multi-objective optimization on the pruned model.
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

class HybridParetoePruningUnlearning(UnlearningStrategy):
    """
    Hybrid unlearning strategy combining dynamic pruning and Pareto optimization.
    
    This method:
    1. Phase 1 - Dynamic Pruning:
       - Computes importance scores for parameters based on forget data
       - Creates pruning masks to isolate forget-sensitive regions
       - Applies selective pruning to reduce forget influence
    
    2. Phase 2 - Pareto Optimization:
       - Performs multi-objective optimization on pruned model
       - Balances forgetting and retention objectives
       - Adaptively adjusts weights to find Pareto optimal solutions
    
    3. Phase 3 - Refinement:
       - Fine-tunes the model with constrained optimization
       - Maintains pruning structure while optimizing trade-offs
    """
    
    def __init__(self, config: FLConfig,
                 # Pruning parameters
                 pruning_ratio: float = 0.15,
                 importance_threshold: float = 0.6,
                 pruning_iterations: int = 3,
                 # Pareto parameters
                 forget_weight: float = 0.5,
                 retention_weight: float = 0.5,
                 pareto_steps: int = 20,
                 adaptive_weights: bool = True,
                 # Hybrid parameters
                 phase1_epochs: int = 5,
                 phase2_epochs: int = 10,
                 refinement_epochs: int = 3,
                 use_gradient_masking: bool = True):
        super().__init__(config)
        
        # Pruning hyperparameters
        self.pruning_ratio = pruning_ratio
        self.importance_threshold = importance_threshold
        self.pruning_iterations = pruning_iterations
        
        # Pareto hyperparameters
        self.forget_weight = forget_weight
        self.retention_weight = retention_weight
        self.pareto_steps = pareto_steps
        self.adaptive_weights = adaptive_weights
        
        # Hybrid hyperparameters
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.refinement_epochs = refinement_epochs
        self.use_gradient_masking = use_gradient_masking
        
        # State tracking
        self.pruning_mask: Dict[str, torch.Tensor] = {}
        self.importance_scores: Dict[str, torch.Tensor] = {}
        self.pareto_frontier: List[Dict[str, Any]] = []
        self.current_weights: Tuple[float, float] = (forget_weight, retention_weight)
        self.phase_metrics: Dict[str, Dict[str, float]] = {
            'phase1': {}, 'phase2': {}, 'phase3': {}
        }
    
    # ==================== Phase 1: Dynamic Pruning ====================
    
    def _compute_parameter_importance(self, model: nn.Module, 
                                     forget_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Compute gradient-based importance scores for each parameter.
        
        Uses accumulated gradient magnitudes on forget data as proxy for
        parameter importance in encoding forget information.
        """
        model.eval()
        importance_scores = {}
        
        # Initialize importance scores
        for name, param in model.named_parameters():
            if param.requires_grad:
                importance_scores[name] = torch.zeros_like(param)
        
        criterion = nn.CrossEntropyLoss()
        num_batches = 0
        
        for data, target in forget_loader:
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Accumulate gradient magnitudes
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    importance_scores[name] += torch.abs(param.grad)
            
            num_batches += 1
        
        # Normalize by number of batches
        for name in importance_scores:
            importance_scores[name] /= max(num_batches, 1)
        
        return importance_scores
    
    def _create_adaptive_pruning_mask(self, importance_scores: Dict[str, torch.Tensor],
                                     iteration: int) -> Dict[str, torch.Tensor]:
        """
        Create adaptive pruning mask with iterative refinement.
        
        Parameters with high importance on forget data are progressively pruned.
        """
        pruning_mask = {}
        
        # Adjust threshold based on iteration
        adjusted_threshold = self.importance_threshold * (1.0 - 0.1 * iteration / self.pruning_iterations)
        adjusted_ratio = self.pruning_ratio * (1.0 + 0.2 * iteration / self.pruning_iterations)
        
        for name, importance in importance_scores.items():
            # Normalize importance scores to [0, 1]
            if importance.max() > importance.min():
                normalized_importance = (importance - importance.min()) / \
                                      (importance.max() - importance.min() + 1e-8)
            else:
                normalized_importance = torch.zeros_like(importance)
            
            # Create mask: prune high-importance parameters (they encode forget data)
            mask = (normalized_importance < adjusted_threshold).float()
            
            # Ensure minimum retention
            num_params = mask.numel()
            num_to_keep = max(1, int(num_params * (1 - adjusted_ratio)))
            
            if mask.sum() < num_to_keep:
                # Keep parameters with lowest importance
                flat_importance = normalized_importance.flatten()
                _, indices = torch.topk(flat_importance, num_to_keep, largest=False)
                mask = torch.zeros_like(normalized_importance).flatten()
                mask[indices] = 1.0
                mask = mask.reshape(normalized_importance.shape)
            
            pruning_mask[name] = mask
        
        return pruning_mask
    
    def _apply_pruning_with_smoothing(self, model: nn.Module, 
                                     pruning_mask: Dict[str, torch.Tensor],
                                     smoothing_factor: float = 0.9):
        """
        Apply pruning with smooth transition to avoid abrupt changes.
        """
        for name, param in model.named_parameters():
            if name in pruning_mask:
                mask = pruning_mask[name].to(param.device)
                # Smooth pruning: gradually reduce pruned weights
                param.data = param.data * (smoothing_factor + (1 - smoothing_factor) * mask)
    
    def _phase1_dynamic_pruning(self, model: nn.Module, 
                               forget_loader: DataLoader,
                               retain_loader: DataLoader):
        """
        Phase 1: Iterative dynamic pruning to isolate forget regions.
        """
        for iteration in range(self.pruning_iterations):
            # Compute importance scores
            self.importance_scores = self._compute_parameter_importance(model, forget_loader)
            
            # Create adaptive pruning mask
            iteration_mask = self._create_adaptive_pruning_mask(self.importance_scores, iteration)
            
            # Merge with existing mask
            if not self.pruning_mask:
                self.pruning_mask = iteration_mask
            else:
                for name in self.pruning_mask:
                    self.pruning_mask[name] *= iteration_mask[name]
            
            # Apply pruning with smoothing
            self._apply_pruning_with_smoothing(model, self.pruning_mask)
            
            # Light fine-tuning on retain data
            self._constrained_fine_tune(model, retain_loader, epochs=2, lr_factor=0.1)
        
        # Evaluate phase 1
        self.phase_metrics['phase1'] = self._evaluate_phase(model, forget_loader, retain_loader)
    
    def _constrained_fine_tune(self, model: nn.Module, retain_loader: DataLoader,
                              epochs: int = 2, lr_factor: float = 0.1):
        """Fine-tune with gradient masking to respect pruning structure."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate * lr_factor)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for data, target in retain_loader:
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply gradient masking
                if self.use_gradient_masking:
                    for name, param in model.named_parameters():
                        if name in self.pruning_mask and param.grad is not None:
                            mask = self.pruning_mask[name].to(param.device)
                            param.grad *= mask
                
                optimizer.step()
    
    # ==================== Phase 2: Pareto Optimization ====================
    def _sample_batch(self, loader: DataLoader):
        """Get one random batch safely — works with any batch_size."""
        data, target = next(iter(loader))
        return data.to(self.config.device), target.to(self.config.device)

    def _compute_multi_objective_loss(self, model: nn.Module,
                                    forget_loader: DataLoader,
                                    retain_loader: DataLoader):
        model.train()
        criterion = nn.CrossEntropyLoss()

        f_data, f_target = self._sample_batch(forget_loader)
        r_data, r_target = self._sample_batch(retain_loader)

        forget_loss = criterion(model(f_data), f_target)
        retain_loss = criterion(model(r_data), r_target)

        total_loss = -self.current_weights[0] * forget_loss + self.current_weights[1] * retain_loss

        return total_loss, {
            'forget_loss': forget_loss.item(),
            'retention_loss': retain_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _update_adaptive_weights(self, forget_acc: float, retain_acc: float,
                                forget_loss: float, retention_loss: float):
        """
        Adaptively update Pareto weights based on current performance.
        """
        if not self.adaptive_weights:
            return
        
        # Compute normalized metrics
        forget_quality = 1.0 - forget_acc  # Higher is better (more forgetting)
        retain_quality = retain_acc  # Higher is better (more retention)
        
        # Adaptive weighting: focus on weaker objective
        if forget_quality < 0.5:  # Poor forgetting
            forget_weight = 0.6 + 0.2 * (1.0 - forget_quality)
        else:  # Good forgetting
            forget_weight = 0.4 - 0.2 * forget_quality
        
        retention_weight = 1.0 - forget_weight
        
        # Smooth transition
        alpha = 0.7
        self.current_weights = (
            alpha * self.current_weights[0] + (1 - alpha) * forget_weight,
            alpha * self.current_weights[1] + (1 - alpha) * retention_weight
        )
    
    def _phase2_pareto_optimization(self, model: nn.Module,
                                forget_loader: DataLoader,
                                retain_loader: DataLoader):
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.5)
        self.pareto_frontier = []

        # 5–8 steps is more than enough for a visible frontier
        for step in range(6):        # was 20 → now 6
            for epoch in range(3):   # was 10 → now 3
                # keep everything else exactly the same
                total_loss, objectives = self._compute_multi_objective_loss(
                    model, forget_loader, retain_loader
                )
                optimizer.zero_grad()
                total_loss.backward()

                if self.use_gradient_masking and self.pruning_mask:
                    for name, param in model.named_parameters():
                        if name in self.pruning_mask and param.grad is not None:
                            param.grad *= self.pruning_mask[name].to(param.device)

                optimizer.step()

                # Evaluate only every few steps (optional but fast)
                if epoch == 2:
                    forget_acc = self._evaluate_accuracy(model, forget_loader)
                    retain_acc = self._evaluate_accuracy(model, retain_loader)
                    self._update_adaptive_weights(forget_acc, retain_acc,
                                                objectives['forget_loss'],
                                                objectives['retention_loss'])

                    self.pareto_frontier.append({
                        'forget_accuracy': forget_acc,
                        'retain_accuracy': retain_acc,
                        'forget_loss': objectives['forget_loss'],
                        'retention_loss': objectives['retention_loss'],
                        'forget_weight': self.current_weights[0],
                        'retention_weight': self.current_weights[1],
                        'step': step
                    })
    # ==================== Phase 3: Refinement ====================
    
    def _phase3_refinement(self, model: nn.Module,
                           forget_loader: DataLoader,
                           retain_loader: DataLoader):
        """Phase 3: Final refinement with balanced objectives."""
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.05)

        # Use the last recorded Pareto point, or fall back to initial weights
        if self.pareto_frontier:
            optimal_point = self._find_pareto_optimal_solution()
            if optimal_point:  # could still be {} if all points were bad
                self.current_weights = (
                    optimal_point.get('forget_weight', self.forget_weight),
                    optimal_point.get('retention_weight', self.retention_weight)
                )

        # If still no frontier (e.g. we skipped too many evals), just use 0.5/0.5
        if sum(self.current_weights) == 0:
            self.current_weights = (0.5, 0.5)

        print(f"Phase 3 using weights: forget={self.current_weights[0]:.3f}, retain={self.current_weights[1]:.3f}")

        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.refinement_epochs):
            total_loss, _ = self._compute_multi_objective_loss(
                model, forget_loader, retain_loader
            )
            optimizer.zero_grad()
            total_loss.backward()

            if self.use_gradient_masking and self.pruning_mask:
                for name, param in model.named_parameters():
                    if name in self.pruning_mask and param.grad is not None:
                        param.grad *= self.pruning_mask[name].to(param.device)

            optimizer.step()

        self.phase_metrics['phase3'] = self._evaluate_phase(model, forget_loader, retain_loader)
    
    # ==================== Utility Methods ====================
    
    def _evaluate_accuracy(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
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
    
    def _evaluate_phase(self, model: nn.Module, forget_loader: DataLoader,
                       retain_loader: DataLoader) -> Dict[str, float]:
        """Evaluate metrics for current phase."""
        forget_acc = self._evaluate_accuracy(model, forget_loader)
        retain_acc = self._evaluate_accuracy(model, retain_loader)
        
        return {
            'forget_accuracy': forget_acc,
            'retain_accuracy': retain_acc,
            'unlearning_effectiveness': 1.0 - forget_acc,
            'retention_preservation': retain_acc
        }
    
    def _find_pareto_optimal_solution(self) -> Dict[str, Any]:
        """
        Find Pareto optimal solution from frontier.
        
        Uses scalarization: minimize weighted distance from ideal point.
        """
        if not self.pareto_frontier:
            return {}
        
        best_idx = 0
        best_score = float('inf')
        
        for i, point in enumerate(self.pareto_frontier):
            # Ideal: high forget effectiveness + high retain accuracy
            forget_eff = 1.0 - point['forget_accuracy']
            retain_acc = point['retain_accuracy']
            
            # Distance from ideal (1, 1)
            score = (1.0 - forget_eff)**2 + (1.0 - retain_acc)**2
            
            if score < best_score:
                best_score = score
                best_idx = i
        
        return self.pareto_frontier[best_idx]
    
    # ==================== Main Unlearning Method ====================
    
    def unlearn(self, model: nn.Module, forget_data: DataLoader,
                retain_data: DataLoader) -> nn.Module:
        """
        Perform hybrid Pareto-pruning unlearning.
        
        Pipeline:
        1. Phase 1: Dynamic pruning to isolate forget regions
        2. Phase 2: Pareto optimization for optimal trade-offs
        3. Phase 3: Refinement with balanced objectives
        """
        torch.cuda.empty_cache()
        print(f"GPU memory before unlearning: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Reduce batch size to 32–64 if still OOM
        # forget_data.batch_size = 16
        # retain_data.batch_size = 16

        unlearned_model = copy.deepcopy(model)
        unlearned_model.to(self.config.device)
        
        # Reset state
        self.pruning_mask = {}
        self.importance_scores = {}
        self.pareto_frontier = []
        self.current_weights = (self.forget_weight, self.retention_weight)
        
        print("Phase 1: Dynamic Pruning...")
        self._phase1_dynamic_pruning(unlearned_model, forget_data, retain_data)
        
        print("Phase 2: Pareto Optimization...")
        self._phase2_pareto_optimization(unlearned_model, forget_data, retain_data)
        
        print("Phase 3: Refinement...")
        self._phase3_refinement(unlearned_model, forget_data, retain_data)
        
        return unlearned_model
    
    def evaluate_unlearning(self, model: nn.Module, forget_data: DataLoader,
                           retain_data: DataLoader) -> Dict[str, float]:
        """Comprehensive evaluation of hybrid unlearning."""
        model.eval()
        device = self.config.device
        criterion = nn.CrossEntropyLoss()
        
        # Evaluate forget data
        forget_correct = 0
        forget_total = 0
        forget_loss = 0.0
        
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
        
        # Evaluate retain data
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
        
        # Pareto optimal solution
        pareto_optimal = self._find_pareto_optimal_solution()
        
        # Pruning statistics
        total_params = sum(mask.numel() for mask in self.pruning_mask.values())
        pruned_params = sum((mask == 0).sum().item() for mask in self.pruning_mask.values())
        
        return {
            # Core metrics
            'forget_accuracy': forget_accuracy,
            'retain_accuracy': retain_accuracy,
            'forget_loss': forget_loss / len(forget_data) if len(forget_data) > 0 else 0.0,
            'retain_loss': retain_loss / len(retain_data) if len(retain_data) > 0 else 0.0,
            'unlearning_effectiveness': 1.0 - forget_accuracy,
            'retention_preservation': retain_accuracy,
            
            # Pruning metrics
            'pruning_ratio': pruned_params / max(total_params, 1),
            'total_parameters': total_params,
            'pruned_parameters': pruned_params,
            'active_parameters': total_params - pruned_params,
            
            # Pareto metrics
            'pareto_frontier_size': len(self.pareto_frontier),
            'pareto_optimal_forget_acc': pareto_optimal.get('forget_accuracy', forget_accuracy),
            'pareto_optimal_retain_acc': pareto_optimal.get('retain_accuracy', retain_accuracy),
            'final_forget_weight': self.current_weights[0],
            'final_retention_weight': self.current_weights[1],
            
            # Phase-wise metrics
            'phase1_forget_acc': self.phase_metrics['phase1'].get('forget_accuracy', 0),
            'phase1_retain_acc': self.phase_metrics['phase1'].get('retain_accuracy', 0),
            'phase2_forget_acc': self.phase_metrics['phase2'].get('forget_accuracy', 0),
            'phase2_retain_acc': self.phase_metrics['phase2'].get('retain_accuracy', 0),
            'phase3_forget_acc': self.phase_metrics['phase3'].get('forget_accuracy', 0),
            'phase3_retain_acc': self.phase_metrics['phase3'].get('retain_accuracy', 0),
            
            # Improvement metrics
            'phase1_to_phase3_forget_improvement': 
                self.phase_metrics['phase3'].get('unlearning_effectiveness', 0) - 
                self.phase_metrics['phase1'].get('unlearning_effectiveness', 0),
            'phase1_to_phase3_retain_preservation':
                self.phase_metrics['phase3'].get('retention_preservation', 0) - 
                self.phase_metrics['phase1'].get('retention_preservation', 0)
        }
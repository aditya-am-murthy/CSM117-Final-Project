"""
Unlearning strategies for federated learning.

This module implements various machine unlearning strategies including
SISA, gradient negation, knowledge distillation, and other methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import copy
import numpy as np
from abc import ABC, abstractmethod

from ..fl.base import UnlearningStrategy, FLConfig


class SISAUnlearning(UnlearningStrategy):
    """
    SISA (Sliced Inverse Regression for Selective Amnesia) unlearning strategy.
    
    This method partitions the model into slices and retrains only the affected slices.
    """
    
    def __init__(self, config: FLConfig, num_slices: int = 4):
        super().__init__(config)
        self.num_slices = num_slices
        self.slice_assignments: Dict[int, int] = {}
    
    def _partition_model(self, model: nn.Module) -> List[nn.Module]:
        """Partition the model into slices."""
        slices = []
        
        for i in range(self.num_slices):
            slice_model = copy.deepcopy(model)
            slices.append(slice_model)
        
        return slices
    
    def _assign_samples_to_slices(self, forget_data: DataLoader):
        """Assign samples to slices."""
        slice_id = 0
        for batch_idx, (data, target) in enumerate(forget_data):
            batch_size = data.size(0)
            for i in range(batch_size):
                sample_id = batch_idx * batch_size + i
                self.slice_assignments[sample_id] = slice_id % self.num_slices
                slice_id += 1
    
    def unlearn(self, model: nn.Module, forget_data: DataLoader,
                retain_data: DataLoader) -> nn.Module:
        """Perform SISA unlearning."""
        self._assign_samples_to_slices(forget_data)
        
        model_slices = self._partition_model(model)
        
        for slice_idx, slice_model in enumerate(model_slices):
            affected_samples = [
                sample_id for sample_id, assigned_slice in self.slice_assignments.items()
                if assigned_slice == slice_idx
            ]
            
            if affected_samples:
                self._retrain_slice(slice_model, retain_data)
        
        return model_slices[0]
    
    def _retrain_slice(self, slice_model: nn.Module, retain_data: DataLoader):
        """Retrain a model slice on retain data."""
        optimizer = optim.SGD(slice_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        slice_model.train()
        for epoch in range(5):
            for data, target in retain_data:
                optimizer.zero_grad()
                output = slice_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    
    def evaluate_unlearning(self, model: nn.Module, forget_data: DataLoader,
                           retain_data: DataLoader) -> Dict[str, float]:
        """Evaluate SISA unlearning effectiveness."""
        model.eval()
        
        forget_correct = 0
        forget_total = 0
        with torch.no_grad():
            for data, target in forget_data:
                output = model(data)
                pred = output.argmax(dim=1)
                forget_correct += (pred == target).sum().item()
                forget_total += target.size(0)
        retain_correct = 0
        retain_total = 0
        with torch.no_grad():
            for data, target in retain_data:
                output = model(data)
                pred = output.argmax(dim=1)
                retain_correct += (pred == target).sum().item()
                retain_total += target.size(0)
        
        return {
            'forget_accuracy': forget_correct / forget_total,
            'retain_accuracy': retain_correct / retain_total,
            'unlearning_effectiveness': 1.0 - (forget_correct / forget_total)
        }


class GradientNegationUnlearning(UnlearningStrategy):
    """
    Gradient negation unlearning strategy.
    
    This method performs gradient ascent on the forget data to "unlearn" it.
    """
    
    def __init__(self, config: FLConfig, unlearning_epochs: int = 10):
        super().__init__(config)
        self.unlearning_epochs = unlearning_epochs
    
    def unlearn(self, model: nn.Module, forget_data: DataLoader,
                retain_data: DataLoader) -> nn.Module:
        """Perform gradient negation unlearning."""
        unlearned_model = copy.deepcopy(model)
        
        optimizer = optim.SGD(unlearned_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        unlearned_model.train()
        
        for epoch in range(self.unlearning_epochs):
            for data, target in forget_data:
                optimizer.zero_grad()
                output = unlearned_model(data)
                loss = criterion(output, target)
                
                (-loss).backward()
                optimizer.step()
        
        return unlearned_model
    
    def evaluate_unlearning(self, model: nn.Module, forget_data: DataLoader,
                           retain_data: DataLoader) -> Dict[str, float]:
        """Evaluate gradient negation unlearning effectiveness."""
        model.eval()
        
        forget_correct = 0
        forget_total = 0
        forget_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in forget_data:
                output = model(data)
                loss = criterion(output, target)
                forget_loss += loss.item()
                
                pred = output.argmax(dim=1)
                forget_correct += (pred == target).sum().item()
                forget_total += target.size(0)
        
        retain_correct = 0
        retain_total = 0
        retain_loss = 0.0
        
        with torch.no_grad():
            for data, target in retain_data:
                output = model(data)
                loss = criterion(output, target)
                retain_loss += loss.item()
                
                pred = output.argmax(dim=1)
                retain_correct += (pred == target).sum().item()
                retain_total += target.size(0)
        
        return {
            'forget_accuracy': forget_correct / forget_total,
            'retain_accuracy': retain_correct / retain_total,
            'forget_loss': forget_loss / len(forget_data),
            'retain_loss': retain_loss / len(retain_data),
            'unlearning_effectiveness': 1.0 - (forget_correct / forget_total)
        }


class KnowledgeDistillationUnlearning(UnlearningStrategy):
    """
    Knowledge distillation unlearning strategy.
    
    This method uses a teacher model (trained without forget data) to guide
    the unlearning process through knowledge distillation.
    """
    
    def __init__(self, config: FLConfig, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__(config)
        self.temperature = temperature
        self.alpha = alpha
    
    def unlearn(self, model: nn.Module, forget_data: DataLoader,
                retain_data: DataLoader) -> nn.Module:
        """Perform knowledge distillation unlearning."""
        teacher_model = copy.deepcopy(model)
        self._train_teacher(teacher_model, retain_data)
        
        student_model = copy.deepcopy(model)
        
        optimizer = optim.SGD(student_model.parameters(), lr=self.config.learning_rate)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
        student_model.train()
        teacher_model.eval()
        
        for epoch in range(10):
            for data, target in retain_data:
                optimizer.zero_grad()
                
                student_output = student_model(data)
                teacher_output = teacher_model(data)
                
                hard_loss = criterion_ce(student_output, target)
                
                soft_loss = criterion_kl(
                    torch.log_softmax(student_output / self.temperature, dim=1),
                    torch.softmax(teacher_output / self.temperature, dim=1)
                ) * (self.temperature ** 2)
                
                total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
                
                total_loss.backward()
                optimizer.step()
        
        return student_model
    
    def _train_teacher(self, teacher_model: nn.Module, retain_data: DataLoader):
        """Train teacher model on retain data only."""
        optimizer = optim.SGD(teacher_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        teacher_model.train()
        for epoch in range(5):
            for data, target in retain_data:
                optimizer.zero_grad()
                output = teacher_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    
    def evaluate_unlearning(self, model: nn.Module, forget_data: DataLoader,
                           retain_data: DataLoader) -> Dict[str, float]:
        """Evaluate knowledge distillation unlearning effectiveness."""
        model.eval()
        
        forget_correct = 0
        forget_total = 0
        with torch.no_grad():
            for data, target in forget_data:
                output = model(data)
                pred = output.argmax(dim=1)
                forget_correct += (pred == target).sum().item()
                forget_total += target.size(0)
        
        retain_correct = 0
        retain_total = 0
        with torch.no_grad():
            for data, target in retain_data:
                output = model(data)
                pred = output.argmax(dim=1)
                retain_correct += (pred == target).sum().item()
                retain_total += target.size(0)
        
        return {
            'forget_accuracy': forget_correct / forget_total,
            'retain_accuracy': retain_correct / retain_total,
            'unlearning_effectiveness': 1.0 - (forget_correct / forget_total)
        }


class FisherInformationUnlearning(UnlearningStrategy):
    """
    Fisher Information Matrix based unlearning strategy.
    
    This method uses the Fisher Information Matrix to identify and modify
    parameters that are most influenced by the forget data.
    """
    
    def __init__(self, config: FLConfig, lambda_reg: float = 0.1):
        super().__init__(config)
        self.lambda_reg = lambda_reg
    
    def unlearn(self, model: nn.Module, forget_data: DataLoader,
                retain_data: DataLoader) -> nn.Module:
        """Perform Fisher Information based unlearning."""
        unlearned_model = copy.deepcopy(model)
        
        fisher_info = self._compute_fisher_information(model, forget_data)
        
        self._modify_parameters(unlearned_model, fisher_info)
        
        return unlearned_model
    
    def _compute_fisher_information(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix."""
        model.eval()
        fisher_info = {}
        
        for name, param in model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)
        
        criterion = nn.CrossEntropyLoss()
        
        for data, target in data_loader:
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad ** 2
        
        num_samples = len(data_loader.dataset)
        for name in fisher_info:
            fisher_info[name] /= num_samples
        
        return fisher_info
    
    def _modify_parameters(self, model: nn.Module, fisher_info: Dict[str, torch.Tensor]):
        """Modify model parameters based on Fisher Information."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in fisher_info:
                    modification = -self.lambda_reg * fisher_info[name] * param
                    param.add_(modification)
    
    def evaluate_unlearning(self, model: nn.Module, forget_data: DataLoader,
                           retain_data: DataLoader) -> Dict[str, float]:
        """Evaluate Fisher Information unlearning effectiveness."""
        model.eval()
        
        forget_correct = 0
        forget_total = 0
        with torch.no_grad():
            for data, target in forget_data:
                output = model(data)
                pred = output.argmax(dim=1)
                forget_correct += (pred == target).sum().item()
                forget_total += target.size(0)
        
        retain_correct = 0
        retain_total = 0
        with torch.no_grad():
            for data, target in retain_data:
                output = model(data)
                pred = output.argmax(dim=1)
                retain_correct += (pred == target).sum().item()
                retain_total += target.size(0)
        
        return {
            'forget_accuracy': forget_correct / forget_total,
            'retain_accuracy': retain_correct / retain_total,
            'unlearning_effectiveness': 1.0 - (forget_correct / forget_total)
        }

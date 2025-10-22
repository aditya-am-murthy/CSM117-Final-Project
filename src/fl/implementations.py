"""
Concrete implementations of federated learning components.

This module provides specific implementations of FL clients and servers,
including FedAvg aggregation and basic client training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any
import random
from .base import FLClient, FLServer, FLConfig


class BasicFLClient(FLClient):
    """Basic implementation of a federated learning client."""
    
    def __init__(self, client_id: int, model: nn.Module, config: FLConfig):
        super().__init__(client_id, model, config)
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_local(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Train the local model for specified number of epochs."""

        self.load_model_state(global_model_state)
        
        self.model.train()
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.local_data):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1

            self.training_history.append({
                'round': self.client_id,  # This will be updated by the server
                'epoch': epoch,
                'loss': epoch_loss / num_batches,
                'client_id': self.client_id
            })
        
        return self.get_model_state()
    
    def evaluate_local(self) -> Dict[str, float]:
        """Evaluate the local model on local data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.local_data:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return {
            'accuracy': correct / total,
            'loss': total_loss / len(self.local_data),
            'data_size': self.data_size
        }


class FedAvgServer(FLServer):
    """Federated Averaging server implementation."""
    
    def __init__(self, global_model: nn.Module, config: FLConfig):
        super().__init__(global_model, config)
        self.client_selection_ratio = 1.0
    
    def aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]], 
                         client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using FedAvg."""
        if not client_updates:
            return self.get_global_model_state()
        
        total_size = sum(client_sizes)
        
        aggregated_params = {}
        for key in client_updates[0].keys():
            aggregated_params[key] = torch.zeros_like(client_updates[0][key])
        
        for client_update, client_size in zip(client_updates, client_sizes):
            weight = client_size / total_size
            for key in aggregated_params.keys():
                aggregated_params[key] += weight * client_update[key]
        
        return aggregated_params
    
    def select_clients(self, round_num: int) -> List[int]:
        """Select clients for the current round."""
        num_clients = len(self.clients)
        num_selected = max(1, int(num_clients * self.client_selection_ratio))
        
        selected_indices = random.sample(range(num_clients), num_selected)
        return selected_indices
    
    def set_client_selection_ratio(self, ratio: float):
        """Set the ratio of clients to select each round."""
        self.client_selection_ratio = max(0.1, min(1.0, ratio))


class FedProxServer(FedAvgServer):
    """FedProx server implementation with proximal term."""
    
    def __init__(self, global_model: nn.Module, config: FLConfig, mu: float = 0.01):
        super().__init__(global_model, config)
        self.mu = mu
    
    def aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]], 
                         client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using FedProx."""
        return super().aggregate_updates(client_updates, client_sizes)


class HeterogeneousFLClient(BasicFLClient):
    """Heterogeneous client with different data distributions."""
    
    def __init__(self, client_id: int, model: nn.Module, config: FLConfig, 
                 data_distribution: str = "iid"):
        super().__init__(client_id, model, config)
        self.data_distribution = data_distribution
        self.class_distribution: Dict[int, float] = {}
    
    def set_class_distribution(self, class_distribution: Dict[int, float]):
        """Set the class distribution for this client."""
        self.class_distribution = class_distribution
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about the client's data."""
        return {
            'client_id': self.client_id,
            'data_size': self.data_size,
            'data_distribution': self.data_distribution,
            'class_distribution': self.class_distribution,
            'num_training_samples': len(self.training_history)
        }

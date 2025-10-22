from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np


@dataclass
class FLConfig:
    num_clients: int = 10
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    aggregation_method: str = "fedavg"
    unlearning_strategy: str = "none"


class FLClient(ABC):
    
    def __init__(self, client_id: int, model: nn.Module, config: FLConfig):
        self.client_id = client_id
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.local_data: Optional[torch.utils.data.DataLoader] = None
        self.data_size: int = 0
        
        self.training_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def train_local(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Train the local model and return updated parameters.
        
        Args:
            global_model_state: Current global model parameters
            
        Returns:
            Updated local model parameters
        """
        pass
    
    @abstractmethod
    def evaluate_local(self) -> Dict[str, float]:
        """
        Evaluate the local model on local data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def set_local_data(self, data_loader: torch.utils.data.DataLoader):
        """Set the local data for this client."""
        self.local_data = data_loader
        self.data_size = len(data_loader.dataset)
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current model state."""
        return self.model.state_dict()
    
    def load_model_state(self, state_dict: Dict[str, torch.Tensor]):
        """Load model state."""
        self.model.load_state_dict(state_dict)


class FLServer(ABC):
    """Abstract base class for federated learning server."""
    
    def __init__(self, global_model: nn.Module, config: FLConfig):
        self.global_model = global_model
        self.config = config
        self.device = torch.device(config.device)
        self.global_model.to(self.device)
        
        # Server state
        self.round: int = 0
        self.clients: List[FLClient] = []
        self.global_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]], 
                         client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates into global model.
        
        Args:
            client_updates: List of client model updates
            client_sizes: List of client data sizes
            
        Returns:
            Aggregated global model parameters
        """
        pass
    
    @abstractmethod
    def select_clients(self, round_num: int) -> List[int]:
        """
        Select clients for the current round.
        
        Args:
            round_num: Current round number
            
        Returns:
            List of selected client IDs
        """
        pass
    
    def add_client(self, client: FLClient):
        """Add a client to the federation."""
        self.clients.append(client)
    
    def get_global_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current global model state."""
        return self.global_model.state_dict()
    
    def evaluate_global(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate global model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return {
            'accuracy': correct / total,
            'loss': total_loss / len(test_loader)
        }


class UnlearningStrategy(ABC):
    """Abstract base class for unlearning strategies."""
    
    def __init__(self, config: FLConfig):
        self.config = config
        
    @abstractmethod
    def unlearn(self, model: nn.Module, forget_data: torch.utils.data.DataLoader,
                retain_data: torch.utils.data.DataLoader) -> nn.Module:
        """
        Perform unlearning on the model.
        
        Args:
            model: Model to unlearn from
            forget_data: Data to forget
            retain_data: Data to retain
            
        Returns:
            Unlearned model
        """
        pass
    
    @abstractmethod
    def evaluate_unlearning(self, model: nn.Module, forget_data: torch.utils.data.DataLoader,
                           retain_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate the effectiveness of unlearning.
        
        Args:
            model: Model to evaluate
            forget_data: Data that should be forgotten
            retain_data: Data that should be retained
            
        Returns:
            Dictionary of unlearning evaluation metrics
        """
        pass


class FLExperiment:
    """Main experiment runner for federated learning with unlearning."""
    
    def __init__(self, config: FLConfig):
        self.config = config
        self.server: Optional[FLServer] = None
        self.unlearning_strategy: Optional[UnlearningStrategy] = None
        
    def setup_experiment(self, server: FLServer, unlearning_strategy: UnlearningStrategy):
        """Setup the experiment with server and unlearning strategy."""
        self.server = server
        self.unlearning_strategy = unlearning_strategy
    
    def run_federated_training(self) -> List[Dict[str, Any]]:
        """Run federated training rounds."""
        if self.server is None:
            raise ValueError("Server not initialized. Call setup_experiment first.")
        
        history = []
        
        for round_num in range(self.config.num_rounds):
            selected_clients = self.server.select_clients(round_num)
            
            client_updates = []
            client_sizes = []
            
            for client_id in selected_clients:
                client = self.server.clients[client_id]
                global_state = self.server.get_global_model_state()
                local_update = client.train_local(global_state)
                client_updates.append(local_update)
                client_sizes.append(client.data_size)
            
            aggregated_params = self.server.aggregate_updates(client_updates, client_sizes)
            self.server.global_model.load_state_dict(aggregated_params)
            
            round_metrics = {
                'round': round_num,
                'selected_clients': selected_clients,
                'num_clients': len(selected_clients)
            }
            
            history.append(round_metrics)
            
        return history
    
    def run_unlearning_experiment(self, forget_data: torch.utils.data.DataLoader,
                                 retain_data: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Run unlearning experiment."""
        if self.unlearning_strategy is None:
            raise ValueError("Unlearning strategy not initialized.")
        
        unlearned_model = self.unlearning_strategy.unlearn(
            self.server.global_model, forget_data, retain_data
        )
        
        unlearning_metrics = self.unlearning_strategy.evaluate_unlearning(
            unlearned_model, forget_data, retain_data
        )
        
        return {
            'unlearned_model': unlearned_model,
            'metrics': unlearning_metrics
        }

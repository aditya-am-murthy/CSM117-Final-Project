"""
Data handling and preprocessing for federated learning experiments.

This module provides utilities for loading, splitting, and distributing datasets
across federated learning clients.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict


class DatasetManager:
    """Manages dataset loading and distribution for FL experiments."""
    
    def __init__(self, dataset_name: str = "cifar10", data_dir: str = "./data"):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.transform_train = None
        self.transform_test = None
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup data transforms for training and testing."""
        if self.dataset_name.lower() == "cifar10":
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            self.transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
    
    def load_dataset(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Load the specified dataset."""
        if self.dataset_name.lower() == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=self.transform_train
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=self.transform_test
            )
        elif self.dataset_name.lower() == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=True, download=True, transform=self.transform_train
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=False, download=True, transform=self.transform_test
            )
        elif self.dataset_name.lower() == "mnist":
            train_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=True, download=True, transform=self.transform_train
            )
            test_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=self.transform_test
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        return train_dataset, test_dataset
    
    def create_iid_split(self, dataset: torch.utils.data.Dataset, num_clients: int, 
                        batch_size: int = 32) -> List[DataLoader]:
        """Create IID data split across clients."""
        total_samples = len(dataset)
        samples_per_client = total_samples // num_clients
        
        split_sizes = [samples_per_client] * num_clients
        split_sizes[-1] += total_samples - sum(split_sizes)
        
        client_datasets = random_split(dataset, split_sizes)
        
        client_loaders = []
        for client_dataset in client_datasets:
            loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            client_loaders.append(loader)
        
        return client_loaders
    
    def create_non_iid_split(self, dataset: torch.utils.data.Dataset, num_clients: int,
                           batch_size: int = 32, alpha: float = 0.5) -> List[DataLoader]:
        """
        Create non-IID data split using Dirichlet distribution.
        
        Args:
            dataset: Training dataset
            num_clients: Number of clients
            batch_size: Batch size for data loaders
            alpha: Dirichlet distribution parameter (smaller = more non-IID)
        """
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        else:
            targets = [dataset[i][1] for i in range(len(dataset))]
        
        num_classes = len(set(targets))
        
        class_to_samples = defaultdict(list)
        for idx, target in enumerate(targets):
            class_to_samples[target].append(idx)
        
        client_class_counts = np.random.dirichlet([alpha] * num_clients, num_classes)
        
        client_samples = [[] for _ in range(num_clients)]
        
        for class_id, samples in class_to_samples.items():
            class_distribution = client_class_counts[class_id]
            num_samples_per_client = [
                int(len(samples) * dist) for dist in class_distribution
            ]
            
            remaining_samples = len(samples) - sum(num_samples_per_client)
            for _ in range(remaining_samples):
                client_idx = np.random.choice(num_clients)
                num_samples_per_client[client_idx] += 1
            
            random.shuffle(samples)
            sample_idx = 0
            
            for client_idx, num_samples in enumerate(num_samples_per_client):
                client_samples[client_idx].extend(
                    samples[sample_idx:sample_idx + num_samples]
                )
                sample_idx += num_samples
        
                client_loaders = []
        for samples in client_samples:
            if samples:
                client_dataset = Subset(dataset, samples)
                loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
                client_loaders.append(loader)
        
        return client_loaders
    
    def create_pathological_split(self, dataset: torch.utils.data.Dataset, num_clients: int,
                                batch_size: int = 32, classes_per_client: int = 2) -> List[DataLoader]:
        """
        Create pathological non-IID split where each client has only a few classes.
        
        Args:
            dataset: Training dataset
            num_clients: Number of clients
            batch_size: Batch size for data loaders
            classes_per_client: Number of classes per client
        """
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        else:
            targets = [dataset[i][1] for i in range(len(dataset))]
        
        num_classes = len(set(targets))
        
        class_to_samples = defaultdict(list)
        for idx, target in enumerate(targets):
            class_to_samples[target].append(idx)
        
        client_samples = [[] for _ in range(num_clients)]
        class_list = list(range(num_classes))
        
        for client_idx in range(num_clients):
            client_classes = random.sample(class_list, min(classes_per_client, len(class_list)))
            
            for class_id in client_classes:
                client_samples[client_idx].extend(class_to_samples[class_id])
        
        client_loaders = []
        for samples in client_samples:
            if samples:
                client_dataset = Subset(dataset, samples)
                loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
                client_loaders.append(loader)
        
        return client_loaders
    
    def get_data_statistics(self, client_loaders: List[DataLoader]) -> List[Dict]:
        """Get statistics about data distribution across clients."""
        stats = []
        
        for client_idx, loader in enumerate(client_loaders):
            class_counts = defaultdict(int)
            total_samples = 0
            
            for _, targets in loader:
                for target in targets:
                    class_counts[target.item()] += 1
                    total_samples += 1
            
            stats.append({
                'client_id': client_idx,
                'total_samples': total_samples,
                'class_distribution': dict(class_counts),
                'num_classes': len(class_counts)
            })
        
        return stats


class UnlearningDataSplitter:
    """Utility for splitting data for unlearning experiments."""
    
    @staticmethod
    def split_for_unlearning(dataset: torch.utils.data.Dataset, 
                           forget_ratio: float = 0.1,
                           test_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split dataset into forget, retain, and test sets.
        
        Args:
            dataset: Dataset to split
            forget_ratio: Ratio of data to forget
            test_ratio: Ratio of data for testing
            
        Returns:
            Tuple of (forget_loader, retain_loader, test_loader)
        """
        total_size = len(dataset)
        test_size = int(total_size * test_ratio)
        forget_size = int((total_size - test_size) * forget_ratio)
        retain_size = total_size - test_size - forget_size
        
        test_dataset, temp_dataset = random_split(
            dataset, [test_size, total_size - test_size]
        )
        forget_dataset, retain_dataset = random_split(
            temp_dataset, [forget_size, retain_size]
        )
        
        forget_loader = DataLoader(forget_dataset, batch_size=32, shuffle=True)
        retain_loader = DataLoader(retain_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return forget_loader, retain_loader, test_loader

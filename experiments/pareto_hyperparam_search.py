"""
Hyperparameter Search for HybridParetoePruningUnlearning.

This script performs comprehensive hyperparameter search for the existing
HybridParetoePruningUnlearning model without modifying the original implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import json
import os
import copy
from datetime import datetime
from itertools import product
import argparse
from typing import Dict, List, Any

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Try importing from the pareto_pruning file
    from src.unlearning.experiments.pareto_optimization.pareto_pruning import (
        HybridParetoePruningUnlearning
    )
    from src.fl.base import FLConfig
except ImportError:
    print("ERROR: Cannot import HybridParetoePruningUnlearning.")
    print("Please ensure the module path is correct.")
    print("You may need to adjust the import statement at the top of this file.")
    exit(1)


# =============================================================================
# Model Definitions (for training base models)
# =============================================================================

class SimpleNN(nn.Module):
    """Simple neural network for MNIST."""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CIFAR10Net(nn.Module):
    """CNN for CIFAR-10."""
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# =============================================================================
# Dataset Loading
# =============================================================================

def load_cifar10(use_subset=True):
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        './data', train=True, download=True, transform=transform_train
    )
    
    if use_subset:
        # Use 5000 samples for faster search
        indices = torch.randperm(len(train_dataset))[:5000].tolist()
        train_dataset = Subset(train_dataset, indices)
        print(f"Using CIFAR-10 subset: {len(train_dataset)} samples")
    
    return train_dataset, CIFAR10Net, 10


def load_mnist(use_subset=True):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    
    if use_subset:
        indices = torch.randperm(len(train_dataset))[:5000].tolist()
        train_dataset = Subset(train_dataset, indices)
        print(f"Using MNIST subset: {len(train_dataset)} samples")
    
    return train_dataset, SimpleNN, 5


def split_data(dataset, forget_ratio=0.2, batch_size=32):
    """Split dataset into forget and retain sets."""
    total_size = len(dataset)
    forget_size = int(total_size * forget_ratio)
    
    indices = torch.randperm(total_size).tolist()
    forget_indices = indices[:forget_size]
    retain_indices = indices[forget_size:]
    
    forget_dataset = Subset(dataset, forget_indices)
    retain_dataset = Subset(dataset, retain_indices)
    
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    
    return forget_loader, retain_loader


# =============================================================================
# Model Training
# =============================================================================

def train_base_model(model, train_loader, config, epochs):
    """Train base model."""
    model.to(config.device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training base model for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(config.device), target.to(config.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
    
    return model


# =============================================================================
# Hyperparameter Search
# =============================================================================

class HybridParetoHyperparameterSearch:
    """
    Hyperparameter search specifically for HybridParetoePruningUnlearning.
    
    Searches over all configurable parameters of the model.
    """
    
    def __init__(self, search_type='random', n_samples=20, output_dir='./hyperparam_results'):
        self.search_type = search_type
        self.n_samples = n_samples
        self.output_dir = output_dir
        self.results = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def define_full_search_space(self) -> Dict[str, List]:
        """
        Complete search space for HybridParetoePruningUnlearning.
        Covers all initialization parameters.
        """
        return {
            # Pruning parameters
            'pruning_ratio': [0.1, 0.15, 0.2, 0.25, 0.3],
            'importance_threshold': [0.4, 0.5, 0.6, 0.7],
            'pruning_iterations': [2, 3, 4, 5],
            
            # Pareto parameters
            'forget_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
            'retention_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
            'pareto_steps': [10, 15, 20, 25],
            'adaptive_weights': [True, False],
            
            # Phase parameters
            'phase1_epochs': [3, 5, 7],
            'phase2_epochs': [8, 10, 12, 15],
            'refinement_epochs': [2, 3, 4],
            
            # Other parameters
            'use_gradient_masking': [True, False],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        }
    
    def define_quick_search_space(self) -> Dict[str, List]:
        """
        Smaller search space for quick testing.
        """
        return {
            'pruning_ratio': [0.15, 0.2],
            'importance_threshold': [0.5, 0.6],
            'pruning_iterations': [2, 3],
            'forget_weight': [0.5, 0.7],
            'retention_weight': [0.3, 0.5],
            'pareto_steps': [10, 15],
            'adaptive_weights': [True],
            'phase1_epochs': [3, 5],
            'phase2_epochs': [8, 10],
            'refinement_epochs': [2, 3],
            'use_gradient_masking': [True],
            'learning_rate': [0.001, 0.01],
        }
    
    def define_focused_search_space(self, focus_area='pruning') -> Dict[str, List]:
        """
        Focused search on specific aspect.
        
        Args:
            focus_area: 'pruning', 'pareto', or 'phases'
        """
        base_config = {
            'pruning_ratio': [0.15],
            'importance_threshold': [0.6],
            'pruning_iterations': [3],
            'forget_weight': [0.5],
            'retention_weight': [0.5],
            'pareto_steps': [15],
            'adaptive_weights': [True],
            'phase1_epochs': [5],
            'phase2_epochs': [10],
            'refinement_epochs': [3],
            'use_gradient_masking': [True],
            'learning_rate': [0.001],
        }
        
        if focus_area == 'pruning':
            base_config.update({
                'pruning_ratio': [0.1, 0.15, 0.2, 0.25, 0.3],
                'importance_threshold': [0.4, 0.5, 0.6, 0.7, 0.8],
                'pruning_iterations': [2, 3, 4, 5, 6],
            })
        elif focus_area == 'pareto':
            base_config.update({
                'forget_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
                'retention_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
                'pareto_steps': [5, 10, 15, 20, 25, 30],
                'adaptive_weights': [True, False],
            })
        elif focus_area == 'phases':
            base_config.update({
                'phase1_epochs': [2, 3, 5, 7, 10],
                'phase2_epochs': [5, 8, 10, 12, 15],
                'refinement_epochs': [1, 2, 3, 4, 5],
                'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            })
        
        return base_config
    
    def generate_configurations(self, search_space: Dict[str, List]) -> List[Dict]:
        """Generate hyperparameter configurations."""
        if self.search_type == 'grid':
            keys = search_space.keys()
            values = search_space.values()
            configs = [dict(zip(keys, v)) for v in product(*values)]
            print(f"Grid Search: {len(configs)} configurations")
        
        elif self.search_type == 'random':
            configs = []
            for _ in range(self.n_samples):
                config = {k: np.random.choice(v) for k, v in search_space.items()}
                configs.append(config)
            print(f"Random Search: {len(configs)} configurations")
        
        else:
            raise ValueError(f"Unknown search_type: {self.search_type}")
        
        return configs
    
    def validate_configuration(self, config: Dict) -> Dict:
        """Validate and fix configuration constraints."""
        # Ensure forget_weight + retention_weight balance
        if 'forget_weight' in config and 'retention_weight' in config:
            total = config['forget_weight'] + config['retention_weight']
            if total > 0:
                config['forget_weight'] = config['forget_weight'] / total
                config['retention_weight'] = config['retention_weight'] / total
        
        # Ensure boolean types
        if 'adaptive_weights' in config:
            config['adaptive_weights'] = bool(config['adaptive_weights'])
        if 'use_gradient_masking' in config:
            config['use_gradient_masking'] = bool(config['use_gradient_masking'])
        
        return config
    
    def create_unlearning_strategy(self, config: Dict, fl_config: FLConfig) -> HybridParetoePruningUnlearning:
        """Create unlearning strategy with given hyperparameters."""
        return HybridParetoePruningUnlearning(
            config=fl_config,
            pruning_ratio=config.get('pruning_ratio', 0.15),
            importance_threshold=config.get('importance_threshold', 0.6),
            pruning_iterations=config.get('pruning_iterations', 3),
            forget_weight=config.get('forget_weight', 0.5),
            retention_weight=config.get('retention_weight', 0.5),
            pareto_steps=config.get('pareto_steps', 20),
            adaptive_weights=config.get('adaptive_weights', True),
            phase1_epochs=config.get('phase1_epochs', 5),
            phase2_epochs=config.get('phase2_epochs', 10),
            refinement_epochs=config.get('refinement_epochs', 3),
            use_gradient_masking=config.get('use_gradient_masking', True)
        )
    
    def evaluate_configuration(self, config: Dict, base_model, forget_loader,
                              retain_loader, fl_config: FLConfig) -> Dict[str, float]:
        """Train and evaluate with given hyperparameters."""
        try:
            print(f"  Creating strategy...")
            
            # Create fresh model copy
            model = copy.deepcopy(base_model)
            
            # Update learning rate in fl_config
            fl_config.learning_rate = config.get('learning_rate', 0.001)
            
            # Create unlearning strategy
            strategy = self.create_unlearning_strategy(config, fl_config)
            
            # Perform unlearning
            print(f"  Running unlearning...")
            unlearned_model = strategy.unlearn(model, forget_loader, retain_loader)
            
            # Evaluate
            print(f"  Evaluating...")
            metrics = strategy.evaluate_unlearning(unlearned_model, forget_loader, retain_loader)
            
            return metrics
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {
                'forget_accuracy': 1.0,
                'retain_accuracy': 0.0,
                'unlearning_effectiveness': 0.0,
                'retention_preservation': 0.0,
                'error': str(e)
            }
    
    def compute_score(self, metrics: Dict[str, float], scoring='balanced') -> float:
        """
        Compute overall score for configuration.
        
        Args:
            scoring: 'balanced', 'forget_priority', 'retain_priority'
        """
        forget_score = metrics.get('unlearning_effectiveness', 0)
        retain_score = metrics.get('retention_preservation', 0)
        
        if scoring == 'balanced':
            return 0.5 * forget_score + 0.5 * retain_score
        elif scoring == 'forget_priority':
            return 0.7 * forget_score + 0.3 * retain_score
        elif scoring == 'retain_priority':
            return 0.3 * forget_score + 0.7 * retain_score
        else:
            return 0.5 * forget_score + 0.5 * retain_score
    
    def run_search(self, dataset='cifar10', use_subset=True, 
                   search_scope='quick', scoring='balanced'):
        """
        Run hyperparameter search.
        
        Args:
            dataset: 'cifar10' or 'mnist'
            use_subset: Use subset of data for faster search
            search_scope: 'quick', 'full', or 'pruning'/'pareto'/'phases'
            scoring: How to score configurations
        """
        print("=" * 70)
        print("HYPERPARAMETER SEARCH FOR HYBRID PARETO-PRUNING UNLEARNING")
        print("=" * 70)
        
        # Load dataset
        print(f"\nLoading {dataset} dataset...")
        if dataset == 'cifar10':
            train_dataset, model_class, train_epochs = load_cifar10(use_subset)
        elif dataset == 'mnist':
            train_dataset, model_class, train_epochs = load_mnist(use_subset)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Split data
        forget_loader, retain_loader = split_data(train_dataset)
        full_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Train base model
        print("\nTraining base model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fl_config = FLConfig(device=device, learning_rate=0.001)
        base_model = model_class()
        base_model = train_base_model(base_model, full_loader, fl_config, train_epochs)
        
        # Define search space
        print("\nDefining search space...")
        if search_scope == 'quick':
            search_space = self.define_quick_search_space()
        elif search_scope == 'full':
            search_space = self.define_full_search_space()
        elif search_scope in ['pruning', 'pareto', 'phases']:
            search_space = self.define_focused_search_space(search_scope)
        else:
            raise ValueError(f"Unknown search_scope: {search_scope}")
        
        # Generate configurations
        configs = self.generate_configurations(search_space)
        
        # Run search
        print("\n" + "=" * 70)
        print(f"STARTING SEARCH - {len(configs)} configurations")
        print("=" * 70)
        
        best_score = -float('inf')
        best_config = None
        best_metrics = None
        
        for idx, config in enumerate(configs):
            print(f"\n[{idx+1}/{len(configs)}] Testing configuration:")
            print(json.dumps(config, indent=2))
            
            # Validate
            config = self.validate_configuration(config)
            
            # Evaluate
            metrics = self.evaluate_configuration(
                config, base_model, forget_loader, retain_loader, fl_config
            )
            
            # Compute score
            score = self.compute_score(metrics, scoring)
            
            # Print results
            print(f"\nResults:")
            print(f"  Forget Accuracy: {metrics.get('forget_accuracy', 0):.4f}")
            print(f"  Retain Accuracy: {metrics.get('retain_accuracy', 0):.4f}")
            print(f"  Unlearning Effectiveness: {metrics.get('unlearning_effectiveness', 0):.4f}")
            print(f"  Retention Preservation: {metrics.get('retention_preservation', 0):.4f}")
            print(f"  Overall Score: {score:.4f}")
            
            # Store
            result = {
                'config': config,
                'metrics': metrics,
                'score': score,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            # Track best
            if score > best_score:
                best_score = score
                best_config = config
                best_metrics = metrics
                print("  *** NEW BEST CONFIGURATION ***")
        
        # Save and summarize
        self.save_results(best_config, best_metrics, best_score)
        self.print_summary(best_config, best_metrics, best_score)
        
        return best_config, best_metrics
    
    def save_results(self, best_config, best_metrics, best_score):
        """Save all results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results
        results_file = os.path.join(self.output_dir, f'hybrid_pareto_search_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump({
                'search_type': self.search_type,
                'n_configurations': len(self.results),
                'best_score': best_score,
                'best_config': best_config,
                'best_metrics': best_metrics,
                'all_results': self.results
            }, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {results_file}")
        
        # Save best config
        best_file = os.path.join(self.output_dir, f'best_hybrid_pareto_config_{timestamp}.json')
        with open(best_file, 'w') as f:
            json.dump({
                'config': best_config,
                'metrics': best_metrics,
                'score': best_score
            }, f, indent=2)
        
        print(f"Best config saved to: {best_file}")
    
    def print_summary(self, best_config, best_metrics, best_score):
        """Print search summary."""
        print("\n" + "=" * 70)
        print("SEARCH COMPLETE")
        print("=" * 70)
        
        print(f"\nTotal configurations evaluated: {len(self.results)}")
        print(f"Best Score: {best_score:.4f}")
        
        print("\nBest Configuration:")
        print(json.dumps(best_config, indent=2))
        
        print("\nBest Metrics:")
        for key, value in best_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Top 5
        print("\nTop 5 Configurations:")
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            print(f"\n{i+1}. Score: {result['score']:.4f}")
            print(f"   Forget Acc: {result['metrics'].get('forget_accuracy', 0):.4f}")
            print(f"   Retain Acc: {result['metrics'].get('retain_accuracy', 0):.4f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for HybridParetoePruningUnlearning'
    )
    parser.add_argument('--search-type', type=str, default='random',
                       choices=['grid', 'random'],
                       help='Type of search')
    parser.add_argument('--n-samples', type=int, default=20,
                       help='Number of random samples')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist'],
                       help='Dataset to use')
    parser.add_argument('--use-subset', action='store_true', default=True,
                       help='Use subset of data')
    parser.add_argument('--search-scope', type=str, default='quick',
                       choices=['quick', 'full', 'pruning', 'pareto', 'phases'],
                       help='Scope of search')
    parser.add_argument('--scoring', type=str, default='balanced',
                       choices=['balanced', 'forget_priority', 'retain_priority'],
                       help='Scoring method')
    parser.add_argument('--output-dir', type=str, default='./hyperparam_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create searcher
    searcher = HybridParetoHyperparameterSearch(
        search_type=args.search_type,
        n_samples=args.n_samples,
        output_dir=args.output_dir
    )
    
    # Run search
    best_config, best_metrics = searcher.run_search(
        dataset=args.dataset,
        use_subset=args.use_subset,
        search_scope=args.search_scope,
        scoring=args.scoring
    )
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
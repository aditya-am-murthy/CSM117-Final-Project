"""
Hyperparameter Search Training Script for Hybrid Pareto-Pruning Unlearning.

This script wraps the training in a loop to search over hyperparameters,
with epochs=3 and batch_size=2, saving results to CSV.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from datetime import datetime
import os
from itertools import product
import csv


# ==================== Configuration ====================

class FLConfig:
    """Federated Learning configuration."""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate=0.001):
        self.device = device
        self.learning_rate = learning_rate


# ==================== Models ====================

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


# ==================== Data Loading ====================

def load_cifar10_data():
    """Load CIFAR-10 dataset."""
    from torchvision import datasets, transforms
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        './data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    )
    
    return train_dataset, test_dataset


def split_dataset(dataset, forget_ratio=0.2, batch_size=2):
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


# ==================== Training ====================

def train_base_model(model, train_loader, config, epochs=3):
    """Train base model."""
    model.to(config.device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    epoch_accuracies = []
    
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
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)
        
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    return model, epoch_losses, epoch_accuracies


def evaluate_accuracy(model, data_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / max(total, 1)


# ==================== Hyperparameter Search ====================

def define_hyperparameter_grid():
    """
    Define hyperparameter search grid.
    
    Returns dict with lists of values to search over.
    """
    return {
        # Pruning parameters
        'pruning_ratio': [0.1, 0.15, 0.2],
        'importance_threshold': [0.5, 0.6, 0.7],
        'pruning_iterations': [2, 3],
        
        # Pareto parameters
        'forget_weight': [0.3, 0.5, 0.7],
        'retention_weight': [0.3, 0.5, 0.7],
        'pareto_steps': [5, 10, 15],
        'adaptive_weights': [True, False],
        
        # Phase parameters
        'phase1_epochs': [2, 3],
        'phase2_epochs': [5, 8],
        'refinement_epochs': [2, 3],
        
        # Training parameters
        'learning_rate': [0.001, 0.01],
        'use_gradient_masking': [True, False],
    }


def generate_hyperparameter_combinations(grid, max_combinations=None):
    """
    Generate all combinations of hyperparameters.
    
    Args:
        grid: Dict of hyperparameter lists
        max_combinations: If set, randomly sample this many combinations
    
    Returns:
        List of hyperparameter dicts
    """
    keys = grid.keys()
    values = grid.values()
    
    # Generate all combinations
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"Total possible combinations: {len(all_combinations)}")
    
    # Sample if too many
    if max_combinations and len(all_combinations) > max_combinations:
        import random
        all_combinations = random.sample(all_combinations, max_combinations)
        print(f"Sampling {max_combinations} random combinations")
    
    return all_combinations


def validate_hyperparameters(params):
    """Ensure hyperparameters are valid."""
    # Normalize forget and retention weights to sum to 1
    total = params['forget_weight'] + params['retention_weight']
    if total > 0:
        params['forget_weight'] /= total
        params['retention_weight'] /= total
    
    return params


# ==================== Main Training Loop ====================

def run_hyperparameter_search(dataset='cifar10', use_subset=True, 
                              max_combinations=50, output_dir='./results'):
    """
    Run hyperparameter search over the training script.
    
    Args:
        dataset: Dataset to use
        use_subset: Use subset of data for faster search
        max_combinations: Maximum number of hyperparameter combinations to try
        output_dir: Directory to save results
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f'hyperparam_search_results_{timestamp}.csv')
    
    print("=" * 80)
    print("HYPERPARAMETER SEARCH FOR HYBRID PARETO-PRUNING UNLEARNING")
    print("=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Epochs: 3 (fixed)")
    print(f"Batch size: 2 (fixed)")
    print(f"Results will be saved to: {csv_filename}")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading dataset...")
    if dataset == 'cifar10':
        train_dataset, test_dataset = load_cifar10_data()
        model_class = CIFAR10Net
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Use subset for faster search
    if use_subset:
        subset_size = min(5000, len(train_dataset))
        indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        train_dataset = Subset(train_dataset, indices)
        print(f"Using subset: {len(train_dataset)} samples")
    
    full_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Generate hyperparameter combinations
    print("\nGenerating hyperparameter combinations...")
    hyperparam_grid = define_hyperparameter_grid()
    hyperparameter_combinations = generate_hyperparameter_combinations(
        hyperparam_grid, max_combinations=max_combinations
    )
    
    print(f"\nWill test {len(hyperparameter_combinations)} configurations")
    print("=" * 80)
    
    # Initialize CSV file with headers
    csv_headers = [
        'run_id', 'timestamp', 'dataset',
        # Hyperparameters
        'pruning_ratio', 'importance_threshold', 'pruning_iterations',
        'forget_weight', 'retention_weight', 'pareto_steps', 'adaptive_weights',
        'phase1_epochs', 'phase2_epochs', 'refinement_epochs',
        'learning_rate', 'use_gradient_masking',
        # Training metrics
        'base_train_loss', 'base_train_accuracy',
        'base_forget_accuracy', 'base_retain_accuracy',
        # Unlearning metrics
        'unlearned_forget_accuracy', 'unlearned_retain_accuracy',
        'unlearning_effectiveness', 'retention_preservation',
        'forget_accuracy_improvement', 'retain_accuracy_change',
        # Additional metrics
        'pruned_parameters', 'total_parameters', 'pruning_ratio_actual',
        'pareto_frontier_size',
        # Phase metrics
        'phase1_forget_acc', 'phase1_retain_acc',
        'phase2_forget_acc', 'phase2_retain_acc',
        'phase3_forget_acc', 'phase3_retain_acc',
        # Status
        'status', 'error_message'
    ]
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
    
    # Run hyperparameter search
    results = []
    
    for run_id, params in enumerate(hyperparameter_combinations):
        print(f"\n{'='*80}")
        print(f"RUN {run_id + 1}/{len(hyperparameter_combinations)}")
        print(f"{'='*80}")
        print("Hyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        try:
            # Validate parameters
            params = validate_hyperparameters(params)
            
            # Create fresh model
            model = model_class()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            config = FLConfig(device=device, learning_rate=params['learning_rate'])
            
            # Train base model
            print("\n--- Training Base Model ---")
            model, train_losses, train_accuracies = train_base_model(
                model, full_loader, config, epochs=3
            )
            
            # Split data
            forget_loader, retain_loader = split_dataset(train_dataset, batch_size=2)
            
            # Evaluate base model
            print("\n--- Evaluating Base Model ---")
            base_forget_acc = evaluate_accuracy(model, forget_loader, device)
            base_retain_acc = evaluate_accuracy(model, retain_loader, device)
            print(f"Base Forget Accuracy: {base_forget_acc*100:.2f}%")
            print(f"Base Retain Accuracy: {base_retain_acc*100:.2f}%")
            
            # Import unlearning strategy
            from pareto_pruning import HybridParetoePruningUnlearning
            
            # Create unlearning strategy with current hyperparameters
            print("\n--- Creating Unlearning Strategy ---")
            unlearning_strategy = HybridParetoePruningUnlearning(
                config=config,
                pruning_ratio=params['pruning_ratio'],
                importance_threshold=params['importance_threshold'],
                pruning_iterations=params['pruning_iterations'],
                forget_weight=params['forget_weight'],
                retention_weight=params['retention_weight'],
                pareto_steps=params['pareto_steps'],
                adaptive_weights=params['adaptive_weights'],
                phase1_epochs=params['phase1_epochs'],
                phase2_epochs=params['phase2_epochs'],
                refinement_epochs=params['refinement_epochs'],
                use_gradient_masking=params['use_gradient_masking']
            )
            
            # Perform unlearning
            print("\n--- Running Unlearning ---")
            unlearned_model = unlearning_strategy.unlearn(
                model=model,
                forget_data=forget_loader,
                retain_data=retain_loader
            )
            
            # Evaluate unlearned model
            print("\n--- Evaluating Unlearned Model ---")
            metrics = unlearning_strategy.evaluate_unlearning(
                model=unlearned_model,
                forget_data=forget_loader,
                retain_data=retain_loader
            )
            
            # Compile results
            result = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset,
                
                # Hyperparameters
                **params,
                
                # Training metrics
                'base_train_loss': train_losses[-1],
                'base_train_accuracy': train_accuracies[-1],
                'base_forget_accuracy': base_forget_acc,
                'base_retain_accuracy': base_retain_acc,
                
                # Unlearning metrics
                'unlearned_forget_accuracy': metrics['forget_accuracy'],
                'unlearned_retain_accuracy': metrics['retain_accuracy'],
                'unlearning_effectiveness': metrics['unlearning_effectiveness'],
                'retention_preservation': metrics['retention_preservation'],
                'forget_accuracy_improvement': base_forget_acc - metrics['forget_accuracy'],
                'retain_accuracy_change': metrics['retain_accuracy'] - base_retain_acc,
                
                # Additional metrics
                'pruned_parameters': metrics.get('pruned_parameters', 0),
                'total_parameters': metrics.get('total_parameters', 0),
                'pruning_ratio_actual': metrics.get('pruning_ratio', 0),
                'pareto_frontier_size': metrics.get('pareto_frontier_size', 0),
                
                # Phase metrics
                'phase1_forget_acc': metrics.get('phase1_forget_acc', 0),
                'phase1_retain_acc': metrics.get('phase1_retain_acc', 0),
                'phase2_forget_acc': metrics.get('phase2_forget_acc', 0),
                'phase2_retain_acc': metrics.get('phase2_retain_acc', 0),
                'phase3_forget_acc': metrics.get('phase3_forget_acc', 0),
                'phase3_retain_acc': metrics.get('phase3_retain_acc', 0),
                
                # Status
                'status': 'success',
                'error_message': ''
            }
            
            print("\n--- Results ---")
            print(f"Forget Accuracy: {base_forget_acc*100:.2f}% → {metrics['forget_accuracy']*100:.2f}%")
            print(f"Retain Accuracy: {base_retain_acc*100:.2f}% → {metrics['retain_accuracy']*100:.2f}%")
            print(f"Unlearning Effectiveness: {metrics['unlearning_effectiveness']*100:.2f}%")
            print(f"Retention Preservation: {metrics['retention_preservation']*100:.2f}%")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset,
                **params,
                'status': 'failed',
                'error_message': str(e)
            }
        
        # Save result to CSV
        results.append(result)
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(result)
        
        print(f"✓ Results saved to CSV")
    
    # Summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)
    print(f"Total runs: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('status') == 'success')}")
    print(f"Failed: {sum(1 for r in results if r.get('status') == 'failed')}")
    print(f"\nResults saved to: {csv_filename}")
    
    # Load results as DataFrame and show best configurations
    df = pd.read_csv(csv_filename)
    df_success = df[df['status'] == 'success']
    
    if len(df_success) > 0:
        print("\n" + "=" * 80)
        print("TOP 5 CONFIGURATIONS BY UNLEARNING EFFECTIVENESS")
        print("=" * 80)
        top5 = df_success.nlargest(5, 'unlearning_effectiveness')
        print(top5[['run_id', 'pruning_ratio', 'forget_weight', 'pareto_steps', 
                    'unlearning_effectiveness', 'retention_preservation']].to_string(index=False))
        
        print("\n" + "=" * 80)
        print("TOP 5 CONFIGURATIONS BY BALANCED SCORE")
        print("=" * 80)
        df_success['balanced_score'] = (
            0.5 * df_success['unlearning_effectiveness'] + 
            0.5 * df_success['retention_preservation']
        )
        top5_balanced = df_success.nlargest(5, 'balanced_score')
        print(top5_balanced[['run_id', 'pruning_ratio', 'forget_weight', 'pareto_steps',
                            'balanced_score']].to_string(index=False))
    
    return csv_filename, results


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter search training script')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'mnist'],
                       help='Dataset to use')
    parser.add_argument('--use-subset', action='store_true', default=True,
                       help='Use subset of data')
    parser.add_argument('--max-combinations', type=int, default=50,
                       help='Maximum number of hyperparameter combinations')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run search
    csv_file, results = run_hyperparameter_search(
        dataset=args.dataset,
        use_subset=args.use_subset,
        max_combinations=args.max_combinations,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Complete! Results saved to: {csv_file}")
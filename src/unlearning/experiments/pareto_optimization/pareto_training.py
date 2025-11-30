"""
Complete example showing how to run the Hybrid Pareto-Pruning Unlearning model.

This script demonstrates:
1. Setting up the environment and data
2. Creating and training a base model
3. Configuring the hybrid unlearning strategy
4. Running the unlearning process
5. Evaluating and visualizing results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


# ==================== Step 1: Define FLConfig (Base Configuration) ====================

class FLConfig:
    """Federated Learning configuration (simplified version)."""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate=0.001):
        self.device = device
        self.learning_rate = learning_rate


# ==================== Step 2: Create a Simple Model ====================

class SimpleNN(nn.Module):
    """Simple neural network for MNIST (grayscale 28x28 images)."""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CIFAR10Net(nn.Module):
    """CNN for CIFAR-10 (color 32x32 images)."""
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        # After 3 pooling layers: 32->16->8->4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Conv block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# ==================== Step 3: Create Synthetic Dataset ====================

def create_synthetic_data(num_samples=1000, input_size=784, num_classes=10):
    """Create synthetic dataset for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def load_real_mnist_data():
    """Load real MNIST dataset for realistic training."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset


def load_cifar10_data():
    """Load CIFAR-10 dataset for realistic training."""
    from torchvision import datasets, transforms
    
    # CIFAR-10 specific transforms with data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load training data
    train_dataset = datasets.CIFAR10(
        './data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        './data',
        train=False,
        download=True,
        transform=transform_test
    )
    
    print(f"CIFAR-10 Classes: {train_dataset.classes}")
    
    return train_dataset, test_dataset


def split_dataset(dataset, forget_ratio=0.2, batch_size=8):
    """
    Split dataset into forget and retain sets.
    
    Args:
        dataset: Full dataset
        forget_ratio: Proportion of data to forget (e.g., 0.2 = 20%)
        batch_size: Batch size for dataloaders
    
    Returns:
        forget_loader, retain_loader
    """
    total_size = len(dataset)
    forget_size = int(total_size * forget_ratio)
    retain_size = total_size - forget_size
    
    # Random split
    indices = torch.randperm(total_size).tolist()
    forget_indices = indices[:forget_size]
    retain_indices = indices[forget_size:]
    
    forget_dataset = Subset(dataset, forget_indices)
    retain_dataset = Subset(dataset, retain_indices)
    
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    
    return forget_loader, retain_loader


# ==================== Step 4: Train Base Model ====================

def train_base_model(model, train_loader, config, epochs=5):
    """Train the base model on full dataset."""
    model.to(config.device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print("Training base model...")
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return model


# ==================== Step 5: Import and Use Hybrid Unlearning ====================

# NOTE: Assuming the HybridParetoePruningUnlearning class is imported
# from your_module import HybridParetoePruningUnlearning

# For demonstration, we'll define a minimal version here:
class HybridUnlearningDemo:
    """Simplified version for demonstration."""
    def __init__(self, config, pruning_ratio=0.15, forget_weight=0.5,
                 retention_weight=0.5, pareto_steps=10):
        self.config = config
        self.pruning_ratio = pruning_ratio
        self.forget_weight = forget_weight
        self.retention_weight = retention_weight
        self.pareto_steps = pareto_steps
    
    def unlearn(self, model, forget_data, retain_data):
        print("\n=== Starting Hybrid Unlearning ===")
        print(f"Pruning Ratio: {self.pruning_ratio}")
        print(f"Forget Weight: {self.forget_weight}")
        print(f"Retention Weight: {self.retention_weight}")
        print(f"Pareto Steps: {self.pareto_steps}")
        
        # This is a placeholder - use the actual implementation
        import copy
        unlearned_model = copy.deepcopy(model)
        
        print("\nPhase 1: Dynamic Pruning...")
        # Actual pruning would happen here
        
        print("Phase 2: Pareto Optimization...")
        # Actual optimization would happen here
        
        print("Phase 3: Refinement...")
        # Actual refinement would happen here
        
        return unlearned_model
    
    def evaluate_unlearning(self, model, forget_data, retain_data):
        model.eval()
        device = self.config.device
        
        # Evaluate forget data
        forget_correct = 0
        forget_total = 0
        with torch.no_grad():
            for data, target in forget_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                forget_correct += (pred == target).sum().item()
                forget_total += target.size(0)
        
        # Evaluate retain data
        retain_correct = 0
        retain_total = 0
        with torch.no_grad():
            for data, target in retain_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                retain_correct += (pred == target).sum().item()
                retain_total += target.size(0)
        
        forget_acc = forget_correct / max(forget_total, 1)
        retain_acc = retain_correct / max(retain_total, 1)
        
        return {
            'forget_accuracy': forget_acc,
            'retain_accuracy': retain_acc,
            'unlearning_effectiveness': 1.0 - forget_acc,
            'retention_preservation': retain_acc
        }


# ==================== Step 6: Main Execution ====================

def main(use_real_data=False, dataset_type='cifar10'):
    """Complete workflow for running hybrid unlearning.
    
    Args:
        use_real_data: If True, uses real dataset. If False, uses synthetic data (faster demo)
        dataset_type: 'mnist' or 'cifar10' when use_real_data=True
    """
    
    # 1. Setup
    print("=" * 60)
    print("HYBRID PARETO-PRUNING UNLEARNING DEMO")
    print("=" * 60)
    
    config = FLConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001
    )
    print(f"\nDevice: {config.device}")
    
    # 2. Create dataset and model
    print("\n--- Creating Dataset ---")
    if use_real_data:
        if dataset_type.lower() == 'cifar10':
            print("Loading CIFAR-10 dataset (color images, will take longer)...")
            train_dataset, test_dataset = load_cifar10_data()
            full_dataset = train_dataset
            model = CIFAR10Net(num_classes=10)
            print(f"Dataset size: {len(full_dataset)}")
            print(f"Model type: CNN with {sum(p.numel() for p in model.parameters()):,} parameters")
            training_epochs = 5  # More epochs for CIFAR-10
        elif dataset_type.lower() == 'mnist':
            print("Loading MNIST dataset (grayscale images)...")
            train_dataset, test_dataset = load_real_mnist_data()
            full_dataset = train_dataset
            model = SimpleNN(input_size=784, hidden_size=128, num_classes=10)
            print(f"Dataset size: {len(full_dataset)}")
            print(f"Model type: Simple NN with {sum(p.numel() for p in model.parameters()):,} parameters")
            training_epochs = 5
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
    else:
        print("Using synthetic data (quick demo)...")
        full_dataset = create_synthetic_data(num_samples=1000, input_size=784, num_classes=10)
        model = SimpleNN(input_size=784, hidden_size=128, num_classes=10)
        print(f"Dataset size: {len(full_dataset)}")
        training_epochs = 5
    
    full_loader = DataLoader(full_dataset, batch_size=8, shuffle=True)
    
    # Split into forget and retain
    forget_loader, retain_loader = split_dataset(
        full_dataset, 
        forget_ratio=0.2,  # 20% of data to forget
        batch_size=8
    )
    print(f"Forget set size: {len(forget_loader.dataset)}")
    print(f"Retain set size: {len(retain_loader.dataset)}")
    
    # 3. Create and train base model
    print("\n--- Training Base Model ---")
    model = train_base_model(model, full_loader, config, epochs=training_epochs)
    
    # 4. Evaluate base model
    print("\n--- Base Model Evaluation ---")
    model.eval()
    base_forget_acc = evaluate_accuracy(model, forget_loader, config.device)
    base_retain_acc = evaluate_accuracy(model, retain_loader, config.device)
    print(f"Base Model - Forget Accuracy: {base_forget_acc*100:.2f}%")
    print(f"Base Model - Retain Accuracy: {base_retain_acc*100:.2f}%")
    
    # 5. Create unlearning strategy
    print("\n--- Initializing Hybrid Unlearning Strategy ---")
    
    # FOR ACTUAL USE, REPLACE WITH:
    # from your_module import HybridParetoePruningUnlearning
    # unlearning_strategy = HybridParetoePruningUnlearning(
    
    unlearning_strategy = HybridUnlearningDemo(
        config=config,
        pruning_ratio=0.15,          # Prune 15% of parameters
        forget_weight=0.5,            # Initial forget objective weight
        retention_weight=0.5,         # Initial retention objective weight
        pareto_steps=10               # Number of Pareto optimization steps
    )
    
    # 6. Perform unlearning
    print("\n--- Running Unlearning Process ---")
    unlearned_model = unlearning_strategy.unlearn(
        model=model,
        forget_data=forget_loader,
        retain_data=retain_loader
    )
    
    # 7. Evaluate unlearned model
    print("\n--- Evaluating Unlearned Model ---")
    metrics = unlearning_strategy.evaluate_unlearning(
        model=unlearned_model,
        forget_data=forget_loader,
        retain_data=retain_loader
    )
    
    # 8. Display results
    print("\n" + "=" * 60)
    print("UNLEARNING RESULTS")
    print("=" * 60)
    print(f"\nForget Accuracy (lower is better):")
    print(f"  Base Model:      {base_forget_acc*100:.2f}%")
    print(f"  Unlearned Model: {metrics['forget_accuracy']*100:.2f}%")
    print(f"  Improvement:     {(base_forget_acc - metrics['forget_accuracy'])*100:.2f}%")
    
    print(f"\nRetain Accuracy (higher is better):")
    print(f"  Base Model:      {base_retain_acc*100:.2f}%")
    print(f"  Unlearned Model: {metrics['retain_accuracy']*100:.2f}%")
    print(f"  Change:          {(metrics['retain_accuracy'] - base_retain_acc)*100:.2f}%")
    
    print(f"\nUnlearning Metrics:")
    print(f"  Unlearning Effectiveness: {metrics['unlearning_effectiveness']*100:.2f}%")
    print(f"  Retention Preservation:   {metrics['retention_preservation']*100:.2f}%")
    
    # 9. Visualize results
    visualize_results(base_forget_acc, base_retain_acc, metrics)
    
    return unlearned_model, metrics


def evaluate_accuracy(model, data_loader, device):
    """Helper function to evaluate accuracy."""
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


def visualize_results(base_forget_acc, base_retain_acc, metrics):
    """Visualize unlearning results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    categories = ['Forget Accuracy\n(lower is better)', 'Retain Accuracy\n(higher is better)']
    base_values = [base_forget_acc * 100, base_retain_acc * 100]
    unlearned_values = [metrics['forget_accuracy'] * 100, metrics['retain_accuracy'] * 100]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, base_values, width, label='Base Model', alpha=0.8)
    axes[0].bar(x + width/2, unlearned_values, width, label='Unlearned Model', alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Model Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Unlearning metrics
    metric_names = ['Unlearning\nEffectiveness', 'Retention\nPreservation']
    metric_values = [
        metrics['unlearning_effectiveness'] * 100,
        metrics['retention_preservation'] * 100
    ]
    
    colors = ['#ff6b6b', '#51cf66']
    axes[1].bar(metric_names, metric_values, color=colors, alpha=0.8)
    axes[1].set_ylabel('Score (%)')
    axes[1].set_title('Unlearning Quality Metrics')
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(metric_values):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('unlearning_results.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'unlearning_results.png'")
    plt.show()


# ==================== Step 7: Advanced Usage Examples ====================

def advanced_example():
    """Advanced usage with custom configurations."""
    
    print("\n" + "=" * 60)
    print("ADVANCED USAGE EXAMPLE")
    print("=" * 60)
    
    config = FLConfig()
    
    # Example 1: Aggressive unlearning (prioritize forgetting)
    print("\n--- Example 1: Aggressive Unlearning ---")
    aggressive_strategy = HybridUnlearningDemo(
        config=config,
        pruning_ratio=0.25,       # More aggressive pruning
        forget_weight=0.7,        # Prioritize forgetting
        retention_weight=0.3,
        pareto_steps=15
    )
    
    # Example 2: Conservative unlearning (prioritize retention)
    print("\n--- Example 2: Conservative Unlearning ---")
    conservative_strategy = HybridUnlearningDemo(
        config=config,
        pruning_ratio=0.1,        # Less aggressive pruning
        forget_weight=0.3,        # Prioritize retention
        retention_weight=0.7,
        pareto_steps=20
    )
    
    # Example 3: Balanced unlearning
    print("\n--- Example 3: Balanced Unlearning ---")
    balanced_strategy = HybridUnlearningDemo(
        config=config,
        pruning_ratio=0.15,       # Moderate pruning
        forget_weight=0.5,        # Equal weights
        retention_weight=0.5,
        pareto_steps=15
    )
    
    print("\nStrategies configured successfully!")
    print("\nTip: Choose strategy based on your priorities:")
    print("  - Aggressive: When forgetting is critical (e.g., privacy)")
    print("  - Conservative: When model performance must be maintained")
    print("  - Balanced: For general use cases")


# ==================== Run the Demo ====================

if __name__ == "__main__":
    # Choose your dataset:
    
    # Option 1: Quick demo with synthetic data (10 seconds)
    # print("Running QUICK DEMO with synthetic data...\n")
    # unlearned_model, metrics = main(use_real_data=False)
    
    # Option 2: Realistic demo with CIFAR-10 (10-15 minutes)
    print("Running REALISTIC DEMO with CIFAR-10 dataset...\n")
    unlearned_model, metrics = main(use_real_data=True, dataset_type='cifar10')
    
    # Option 3: Realistic demo with MNIST (5-7 minutes)
    # print("Running REALISTIC DEMO with MNIST dataset...\n")
    # unlearned_model, metrics = main(use_real_data=True, dataset_type='mnist')
    
    # Run advanced examples
    # advanced_example()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
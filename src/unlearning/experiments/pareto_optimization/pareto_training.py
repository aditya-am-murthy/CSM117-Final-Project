"""
Hyperparameter Search Training Script for Hybrid Pareto-Pruning Unlearning
using google/vit-base-patch16-224 (Vision Transformer)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
import os
from itertools import product
import csv
import random
import warnings
warnings.filterwarnings("ignore")

# ==================== Configuration ====================
class FLConfig:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', learning_rate=0.001):
        self.device = device
        self.learning_rate = learning_rate

# ==================== ViT Model Loading ====================
def load_vit_model():
    from transformers import ViTForImageClassification

    print("Loading google/vit-base-patch16-224 safely...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,   # <--- FORCE float32 to avoid NaN
    )

    # Only enable fp16 if you're 100% sure (use with gradient scaler)
    if torch.cuda.is_available():
        model = model.to("cuda")
        # Optional: use scaler only if you want speed
        # from torch.cuda.amp import autocast, GradScaler
        # But for now: keep it simple and stable
    return model
# ==================== Data Loading (ViT-compatible) ====================
def load_cifar10_data():
    """Load CIFAR-10 with ViT preprocessing."""
    from torchvision import datasets, transforms
    from transformers import ViTImageProcessor

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    size = processor.size['height']

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        ),
    ])

    train_dataset = datasets.CIFAR10(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset

def split_dataset(dataset, forget_ratio=0.2, batch_size=16):
    """Split into forget/retain sets."""
    total_size = len(dataset)
    forget_size = int(total_size * forget_ratio)
    indices = torch.randperm(total_size).tolist()
    forget_indices = indices[:forget_size]
    retain_indices = indices[forget_size:]

    forget_dataset = Subset(dataset, forget_indices)
    retain_dataset = Subset(dataset, retain_indices)

    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return forget_loader, retain_loader

# ==================== Training & Evaluation ====================
def train_base_model(model, train_loader, config, epochs=2):
    """Safely train ViT on CIFAR-10 without NaN."""
    model.to(config.device)
    model.train()

    # CRITICAL: Very small learning rate for pretrained ViT
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    
    # Gradient clipping + warm-up style safety
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"Training ViT safely (LR={optimizer.param_groups[0]['lr']}) on {config.device}...")

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(config.device, non_blocking=True)
            targets = targets.to(config.device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            logits = outputs.logits

            loss = criterion(logits, targets)

            # === SAFETY NET ===
            if not torch.isfinite(loss):
                print(f"Loss is NaN/inf at step {step}, skipping...")
                continue

            loss.backward()

            # CRITICAL: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)

            if step % 50 == 0:
                current_acc = 100. * correct / total if total > 0 else 0
                print(f"  [Epoch {epoch+1}/{epochs}] Step {step:3d} | "
                      f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} finished → Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    return model, epoch_loss, epoch_acc

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            pred = logits.argmax(dim=-1)

            correct += (pred == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)

# ==================== Hyperparameter Grid ====================
def define_hyperparameter_grid():
    return {
        'pruning_ratio': [0.1, 0.2, 0.3],
        'importance_threshold': [0.5, 0.6],
        'pruning_iterations': [2, 3],
        'forget_weight': [0.3, 0.5, 0.7],
        'retention_weight': [0.3, 0.5, 0.7],
        'pareto_steps': [6],  # Reduced for speed
        'adaptive_weights': [True, False],
        'phase1_epochs': [3],
        'phase2_epochs': [3],
        'refinement_epochs': [2],
        'learning_rate': [3e-5, 5e-5, 1e-4],
        'use_gradient_masking': [True],
    }

def generate_hyperparameter_combinations(grid, max_combinations=30):
    keys = grid.keys()
    values = grid.values()
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    print(f"Total possible combinations: {len(all_combinations)}")
    if max_combinations and len(all_combinations) > max_combinations:
        all_combinations = random.sample(all_combinations, max_combinations)
        print(f"Sampling {max_combinations} random combinations")
    return all_combinations

def validate_hyperparameters(params):
    total = params['forget_weight'] + params['retention_weight']
    if total > 0:
        params['forget_weight'] /= total
        params['retention_weight'] /= total
    return params

# ==================== Main Search Loop ====================
def run_hyperparameter_search(
    dataset='cifar10',
    use_subset=True,
    max_combinations=30,
    output_dir='./results_vit'
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f'vit_hyperparam_search_{timestamp}.csv')

    print("=" * 90)
    print("HYBRID PARETO-PRUNING UNLEARNING HYPERPARAMETER SEARCH")
    print("MODEL: google/vit-base-patch16-224")
    print("=" * 90)

    # Load data
    train_dataset, _ = load_cifar10_data()
    if use_subset:
        subset_size = 2500  # ViT is heavy
        indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        train_dataset = Subset(train_dataset, indices)
        print(f"Using subset of {len(train_dataset)} samples")

    full_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    # Generate combinations
    hyperparam_grid = define_hyperparameter_grid()
    combinations = generate_hyperparameter_combinations(hyperparam_grid, max_combinations)

    # CSV setup
    headers = [
        'run_id', 'timestamp', 'pruning_ratio', 'importance_threshold', 'pruning_iterations',
        'forget_weight', 'retention_weight', 'pareto_steps', 'adaptive_weights',
        'phase1_epochs', 'phase2_epochs', 'refinement_epochs', 'learning_rate',
        'use_gradient_masking', 'base_forget_accuracy', 'base_retain_accuracy',
        'unlearned_forget_accuracy', 'unlearned_retain_accuracy',
        'unlearning_effectiveness', 'retention_preservation',
        'forget_accuracy_drop', 'retain_accuracy_change',
        'pruned_parameters', 'total_parameters', 'pruning_ratio_actual',
        'pareto_frontier_size', 'status', 'error_message'
    ]

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    results = []

    for run_id, params in enumerate(combinations):
        print(f"\n{'='*90}")
        print(f"RUN {run_id + 1}/{len(combinations)} | ViT Hybrid Unlearning")
        print(f"{'='*90}")
        for k, v in params.items():
            print(f"  {k}: {v}")

        try:
            params = validate_hyperparameters(params)

            # Load fresh ViT model
            model = load_vit_model()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            config = FLConfig(device=device, learning_rate=params['learning_rate'])
            model.to(device)

            # Train base model briefly
            model, _, base_acc = train_base_model(model, full_loader, config, epochs=2)

            # Split data
            forget_loader, retain_loader = split_dataset(train_dataset, batch_size=16)

            # Evaluate base
            base_forget_acc = evaluate_accuracy(model, forget_loader, device)
            base_retain_acc = evaluate_accuracy(model, retain_loader, device)
            print(f"Base → Forget: {base_forget_acc*100:.2f}%, Retain: {base_retain_acc*100:.2f}%")

            # Import and run unlearning
            from .pareto_pruning import HybridParetoePruningUnlearning

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

            unlearned_model = unlearning_strategy.unlearn(
                model=model,
                forget_data=forget_loader,
                retain_data=retain_loader
            )

            metrics = unlearning_strategy.evaluate_unlearning(
                model=unlearned_model,
                forget_data=forget_loader,
                retain_data=retain_loader
            )

            result = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                **params,
                'base_forget_accuracy': base_forget_acc,
                'base_retain_accuracy': base_retain_acc,
                'unlearned_forget_accuracy': metrics['forget_accuracy'],
                'unlearned_retain_accuracy': metrics['retain_accuracy'],
                'unlearning_effectiveness': metrics['unlearning_effectiveness'],
                'retention_preservation': metrics['retention_preservation'],
                'forget_accuracy_drop': base_forget_acc - metrics['forget_accuracy'],
                'retain_accuracy_change': metrics['retain_accuracy'] - base_retain_acc,
                'pruned_parameters': metrics.get('pruned_parameters', 0),
                'total_parameters': metrics.get('total_parameters', 0),
                'pruning_ratio_actual': metrics.get('pruning_ratio', 0),
                'pareto_frontier_size': metrics.get('pareto_frontier_size', 0),
                'status': 'success',
                'error_message': ''
            }

            print(f"Unlearned → Forget: {metrics['forget_accuracy']*100:.2f}%, "
                  f"Retain: {metrics['retain_accuracy']*100:.2f}%")
            print(f"Unlearning Effectiveness: {metrics['unlearning_effectiveness']*100:.2f}%")
            print(f"Retention Preservation: {metrics['retention_preservation']*100:.2f}%")

        except Exception as e:
            print(f"RUN FAILED: {e}")
            import traceback
            traceback.print_exc()
            result = {k: params.get(k.split('_', 1)[1] if '_' in k else k, '') for k in headers if k in params or k in ['run_id', 'timestamp', 'status', 'error_message']}
            result.update({'run_id': run_id, 'timestamp': datetime.now().isoformat(), 'status': 'failed', 'error_message': str(e)})

        # Save result
        results.append(result)
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(result)

    # Final summary
    print("\n" + "="*90)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"Results saved to: {csv_filename}")
    df = pd.read_csv(csv_filename)
    success = df[df['status'] == 'success']
    if len(success) > 0:
        print(f"\nTop 5 by Unlearning Effectiveness:")
        top = success.nlargest(5, 'unlearning_effectiveness')
        print(top[['run_id', 'pruning_ratio', 'forget_weight', 'unlearning_effectiveness', 'retention_preservation']])
    return csv_filename

# ==================== Entry Point ====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-combinations', type=int, default=25)
    parser.add_argument('--output-dir', type=str, default='./results_vit')
    args = parser.parse_args()

    csv_file = run_hyperparameter_search(
        max_combinations=args.max_combinations,
        output_dir=args.output_dir
    )
    print(f"\nAll done! Full results: {csv_file}")
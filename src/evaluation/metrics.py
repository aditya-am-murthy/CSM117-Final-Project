import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class UnlearningEvaluator:
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.metrics_history: List[Dict[str, Any]] = []
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'loss': total_loss / len(test_loader),
            'num_samples': len(all_targets)
        }
    
    def evaluate_unlearning_effectiveness(self, original_model: nn.Module, 
                                       unlearned_model: nn.Module,
                                       forget_loader: DataLoader,
                                       retain_loader: DataLoader,
                                       test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of unlearning.
        
        Args:
            original_model: Model before unlearning
            unlearned_model: Model after unlearning
            forget_loader: Data that should be forgotten
            retain_loader: Data that should be retained
            test_loader: General test data
            
        Returns:
            Comprehensive evaluation metrics
        """
        results = {}
        
        results['original_forget'] = self.evaluate_model(original_model, forget_loader)
        results['original_retain'] = self.evaluate_model(original_model, retain_loader)
        results['original_test'] = self.evaluate_model(original_model, test_loader)
        
        results['unlearned_forget'] = self.evaluate_model(unlearned_model, forget_loader)
        results['unlearned_retain'] = self.evaluate_model(unlearned_model, retain_loader)
        results['unlearned_test'] = self.evaluate_model(unlearned_model, test_loader)
        
        forget_accuracy_drop = (results['original_forget']['accuracy'] - 
                               results['unlearned_forget']['accuracy'])
        retain_accuracy_drop = (results['original_retain']['accuracy'] - 
                               results['unlearned_retain']['accuracy'])
        test_accuracy_drop = (results['original_test']['accuracy'] - 
                             results['unlearned_test']['accuracy'])
        
        results['unlearning_metrics'] = {
            'forget_accuracy_drop': forget_accuracy_drop,
            'retain_accuracy_drop': retain_accuracy_drop,
            'test_accuracy_drop': test_accuracy_drop,
            'forget_effectiveness': forget_accuracy_drop,
            'retention_preservation': 1.0 - retain_accuracy_drop,
            'generalization_preservation': 1.0 - test_accuracy_drop
        }
        
        return results
    
    def evaluate_federated_learning(self, server, test_loader: DataLoader, 
                                  round_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate federated learning performance over rounds.
        
        Args:
            server: FL server with global model
            test_loader: Test data loader
            round_history: History of training rounds
            
        Returns:
            FL evaluation metrics
        """
        current_metrics = self.evaluate_model(server.global_model, test_loader)
        
        round_accuracies = []
        round_losses = []
        client_participation = defaultdict(int)
        
        for round_info in round_history:
            if 'accuracy' in round_info:
                round_accuracies.append(round_info['accuracy'])
            if 'loss' in round_info:
                round_losses.append(round_info['loss'])
            
            if 'selected_clients' in round_info:
                for client_id in round_info['selected_clients']:
                    client_participation[client_id] += 1
        
        fl_metrics = {
            'final_accuracy': current_metrics['accuracy'],
            'final_loss': current_metrics['loss'],
            'convergence_round': self._find_convergence_round(round_accuracies),
            'client_participation_fairness': self._calculate_participation_fairness(client_participation),
            'round_accuracies': round_accuracies,
            'round_losses': round_losses
        }
        
        return fl_metrics
    
    def _find_convergence_round(self, accuracies: List[float], 
                              threshold: float = 0.01, window: int = 5) -> int:
        """Find the round where the model converged."""
        if len(accuracies) < window:
            return len(accuracies)
        
        for i in range(window, len(accuracies)):
            recent_accs = accuracies[i-window:i]
            if max(recent_accs) - min(recent_accs) < threshold:
                return i - window
        
        return len(accuracies)
    
    def _calculate_participation_fairness(self, participation: Dict[int, int]) -> float:
        """Calculate fairness of client participation."""
        if not participation:
            return 0.0
        
        participation_counts = list(participation.values())
        mean_participation = np.mean(participation_counts)
        
        if mean_participation == 0:
            return 0.0
        
        std_participation = np.std(participation_counts)
        cv = std_participation / mean_participation
        
        fairness = 1.0 / (1.0 + cv)
        return fairness
    
    def measure_computational_cost(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Measure computational cost of a function.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (function result, cost metrics)
        """
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        cost_metrics = {
            'execution_time': execution_time,
            'memory_usage': 0.0
        }
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            cost_metrics['memory_usage'] = (end_memory - start_memory) / (1024 ** 2)
        
        return result, cost_metrics
    
    def create_evaluation_report(self, results: Dict[str, Any], 
                               save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 60)
        report.append("FEDERATED LEARNING UNLEARNING EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        if 'unlearning_metrics' in results:
            metrics = results['unlearning_metrics']
            report.append("UNLEARNING EFFECTIVENESS:")
            report.append("-" * 30)
            report.append(f"Forget Accuracy Drop: {metrics['forget_accuracy_drop']:.4f}")
            report.append(f"Retain Accuracy Drop: {metrics['retain_accuracy_drop']:.4f}")
            report.append(f"Test Accuracy Drop: {metrics['test_accuracy_drop']:.4f}")
            report.append(f"Forget Effectiveness: {metrics['forget_effectiveness']:.4f}")
            report.append(f"Retention Preservation: {metrics['retention_preservation']:.4f}")
            report.append(f"Generalization Preservation: {metrics['generalization_preservation']:.4f}")
            report.append("")
        
        if 'unlearned_test' in results:
            test_metrics = results['unlearned_test']
            report.append("MODEL PERFORMANCE:")
            report.append("-" * 20)
            report.append(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            report.append(f"Test Precision: {test_metrics['precision']:.4f}")
            report.append(f"Test Recall: {test_metrics['recall']:.4f}")
            report.append(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
            report.append(f"Test Loss: {test_metrics['loss']:.4f}")
            report.append("")
        
        if 'computational_cost' in results:
            cost = results['computational_cost']
            report.append("COMPUTATIONAL COST:")
            report.append("-" * 20)
            report.append(f"Execution Time: {cost['execution_time']:.2f} seconds")
            report.append(f"Memory Usage: {cost['memory_usage']:.2f} MB")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_training_curves(self, round_history: List[Dict[str, Any]], 
                           save_path: Optional[str] = None):
        """Plot training curves for federated learning."""
        if not round_history:
            return
        
        rounds = list(range(len(round_history)))
        accuracies = [round_info.get('accuracy', 0) for round_info in round_history]
        losses = [round_info.get('loss', 0) for round_info in round_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(rounds, accuracies, 'b-', linewidth=2)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Federated Learning Accuracy')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(rounds, losses, 'r-', linewidth=2)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.set_title('Federated Learning Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_unlearning_comparison(self, results: Dict[str, Any], 
                                 save_path: Optional[str] = None):
        """Plot comparison between original and unlearned models."""
        if 'unlearning_metrics' not in results:
            return
        
        metrics = results['unlearning_metrics']
        
        categories = ['Forget\nAccuracy Drop', 'Retain\nAccuracy Drop', 'Test\nAccuracy Drop']
        values = [
            metrics['forget_accuracy_drop'],
            metrics['retain_accuracy_drop'],
            metrics['test_accuracy_drop']
        ]
        
        colors = ['red', 'blue', 'green']
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(categories, values, color=colors, alpha=0.7)
        
        plt.ylabel('Accuracy Drop')
        plt.title('Unlearning Effectiveness Comparison')
        plt.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

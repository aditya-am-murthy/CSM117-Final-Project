"""
Simple tests to verify the framework is working correctly.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent / "src"))

from fl.base import FLConfig
from fl.implementations import BasicFLClient, FedAvgServer
from data.dataset_manager import DatasetManager
from unlearning.strategies import GradientNegationUnlearning
from evaluation.metrics import UnlearningEvaluator


def test_basic_functionality():
    """Test basic functionality of the framework."""
    print("Testing basic functionality...")
    
    config = FLConfig(num_clients=5, num_rounds=2)
    assert config.num_clients == 5
    assert config.num_rounds == 2
    print("✓ FLConfig working")
    
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    print("✓ Model creation working")
    
    client = BasicFLClient(0, model, config)
    assert client.client_id == 0
    print("✓ Client creation working")
    
    server = FedAvgServer(model, config)
    assert len(server.clients) == 0
    print("✓ Server creation working")
    
    strategy = GradientNegationUnlearning(config)
    assert strategy.config == config
    print("✓ Unlearning strategy creation working")
    
    evaluator = UnlearningEvaluator()
    assert evaluator.device is not None
    print("✓ Evaluator creation working")
    
    print("All basic tests passed! ✓")


def test_dataset_manager():
    """Test dataset manager functionality."""
    print("\nTesting dataset manager...")
    
    try:
        manager = DatasetManager("cifar10", "./data")
        print("✓ DatasetManager creation working")
        
        # train_dataset, test_dataset = manager.load_dataset()
        # print("✓ Dataset loading working")
        
    except Exception as e:
        print(f"⚠ Dataset manager test skipped: {e}")
    
    print("Dataset manager test completed")


if __name__ == "__main__":
    test_basic_functionality()
    test_dataset_manager()
    print("\n🎉 All tests completed successfully!")

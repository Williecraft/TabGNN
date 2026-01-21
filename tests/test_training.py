import os
import torch

def test_training_config_exists():
    assert os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train_model.py"))


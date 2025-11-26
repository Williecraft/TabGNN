import os
import torch

def test_retrieval_module_importable():
    import importlib
    m = importlib.import_module('retrieval')
    assert hasattr(m, 'DiffusionModel')


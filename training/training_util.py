"""Utilities for model development scripts: training and staging."""
import argparse
import importlib

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

DATA_CLASS_MODULE = "data"
MODEL_CLASS_MODULE = "models"


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'module.submodule.MyClass'."""
    module_str, _, class_str = module_and_class_name.rpartition('.')
    module = importlib.import_module(module_str)
    
    # Traverse the module hierarchy to find the correct class
    class_obj = module
    for name in class_str.split('.'):
        class_obj = getattr(class_obj, name, None)
        if class_obj is None:
            raise AttributeError(f"Class '{class_str}' not found in module '{module_str}'.")
    
    return class_obj


def setup_data_and_model_from_args(args: argparse.Namespace):
    data_class = import_class(f"{DATA_CLASS_MODULE}.{args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{args.model_class}")

    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    return data, model
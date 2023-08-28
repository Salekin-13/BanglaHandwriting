"""Models for character and text recognition in images."""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from models.mlp import MLP

from models.cnn import CNN

from models.resnet_transformer import ResnetTransformer
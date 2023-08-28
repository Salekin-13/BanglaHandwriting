import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

import metadata.BanglaEMNIST as bemnist
import metadata.shared as shared


PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed_data" / "B_IAM_paragraphs"

NEW_LINE_TOKEN = "\n"

MAPPING = [*bemnist.MAPPING, NEW_LINE_TOKEN]

# must match IMAGE_SCALE_FACTOR for IAMLines to be compatible with synthetic paragraphs
IMAGE_SCALE_FACTOR = 2
IMAGE_HEIGHT, IMAGE_WIDTH = 1000, 1000
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

MAX_LABEL_LENGTH = 700

DIMS = (1, IMAGE_HEIGHT, IMAGE_WIDTH)
OUTPUT_DIMS = (MAX_LABEL_LENGTH, 1)
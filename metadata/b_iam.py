import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

import metadata.shared as shared

RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "zip" 
ZIP_FILENAME = RAW_DATA_DIRNAME / "B_IAM.zip"
IMAGE_DATA_DIRNAME = shared.DATA_DIRNAME / "raw_images" 
EXTRACTED_DATASET_DIRNAME = IMAGE_DATA_DIRNAME / "B_IAM"

###LINE_REGION_PADDING = 8  # add this many pixels around the exact coordinates
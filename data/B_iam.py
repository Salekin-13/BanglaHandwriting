from pathlib import Path
from typing import Any, Dict, List
import zipfile
import argparse

from boltons.cacheutils import cachedproperty
import json
from PIL import ImageOps

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

import util
from data.base_data_module import BaseDataModule,load_and_print_info
from metadata.b_iam_paragraphs import NEW_LINE_TOKEN
import metadata.b_iam as metadata

ZIP_FILENAME = metadata.ZIP_FILENAME
EXTRACTED_DATASET_DIRNAME = metadata.EXTRACTED_DATASET_DIRNAME
###LINE_PADDING = metadata.LINE_REGION_PADDING

class BIAM(BaseDataModule):
    def __init__(self,args: argparse.Namespace = None):
        super().__init__(args)
    
    def prepare_data(self):
        extracted_data_dir = Path(EXTRACTED_DATASET_DIRNAME)
        json_files_exist = len(list((Path(EXTRACTED_DATASET_DIRNAME) / "json").glob("*.json"))) > 0
        if not extracted_data_dir.exists() or not json_files_exist:
            # Create necessary directories before extraction
            os.makedirs(extracted_data_dir, exist_ok=True)
            with zipfile.ZipFile(ZIP_FILENAME, "r") as zip_file:
                zip_file.extractall(extracted_data_dir)
                
    def setup(self):
        self.prepare_data()            
                
    def load_image(self, id):
        image_path = Path(EXTRACTED_DATASET_DIRNAME) / "forms" / f"{id}.jpg"
        image = util.read_image_pil(image_uri=image_path,grayscale=True)
        image = ImageOps.invert(image)
        return image
    
    def __repr__(self):
        """Print info about the dataset."""
        info = ["BIAM Dataset"]
        info.append(f"Total Images: {len(self.json_filenames)}")
        info.append(f"Total Test Images: {len(self.test_ids)}")
        info.append(f"Total Paragraphs: {len(self.paragraph_string_by_id)}")
        num_lines = sum(len(line_regions) for line_regions in self.line_regions_by_id.values())
        info.append(f"Total Lines: {num_lines}")

        # Return the joined string representation
        return "\n\t".join(info)
    
    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser
    
    @cachedproperty
    def all_ids(self):
        """A list of all form IDs from JSON files."""
        json_filenames = list((Path(EXTRACTED_DATASET_DIRNAME) / "json").glob("*.json"))
        return sorted([f.stem for f in json_filenames]) 
    
    @cachedproperty
    def ids_by_split(self):
        """A dictionary mapping split names to form IDs in each split."""
        return {
            "train": self.train_ids,
            "val": self.validation_ids,
            "test": self.test_ids
        }

    @cachedproperty
    def split_by_id(self):
        """A dictionary mapping form IDs to their split according to the dataset."""
        split_by_id = {}
        for split_name, form_ids in self.ids_by_split.items():
            split_by_id.update({id_: split_name for id_ in form_ids})
        return split_by_id
    
    @cachedproperty
    def train_ids(self):
        """A list of form IDs which are in the training set."""
        return _read_split_ids("train_ids")

    @cachedproperty
    def test_ids(self):
        """A list of form IDs from the test set."""
        return _read_split_ids("test_ids")

    @cachedproperty
    def validation_ids(self):
        """A list of form IDs from the validation set."""
        return _read_split_ids("val_ids")
    
    @property
    def json_filenames(self):
        """A list of the filenames of all .json files, which contain label information."""
        return list((Path(EXTRACTED_DATASET_DIRNAME) / "json").glob("*.json"))

    @property
    def json_filenames_by_id(self):
        """A dictionary mapping form IDs to their JSON label information files."""
        return {filename.stem: filename for filename in self.json_filenames}
    
    @property
    def form_filenames(self):
        """A list of the filenames of all .jpg files, which contain images of B_IAM forms."""
        return list((Path(EXTRACTED_DATASET_DIRNAME) / "forms").glob("*.jpg"))
    
    @property
    def form_filenames_by_id(self):
        """A dictionary mapping form IDs to their JPEG images."""
        form_filenames = list((Path(EXTRACTED_DATASET_DIRNAME) / "forms").glob("*.jpg"))
        return {filename.stem: filename for filename in form_filenames}
    
    @cachedproperty
    def line_strings_by_id(self):
        """A dict mapping an BIAM form id to its list of line texts."""
        return {filename.stem: _get_line_strings_from_json_file(filename) for filename in self.json_filenames}
    
    @cachedproperty
    def paragraph_string_by_id(self):
        """A dict mapping a BIAM form id to its paragraph text."""
        return {id_: NEW_LINE_TOKEN.join(line_strings) for id_, line_strings in self.line_strings_by_id.items()}

    
def _read_split_ids(split_name):
        """Read form IDs for the specified split from the text file."""
        split_ids_file = Path(EXTRACTED_DATASET_DIRNAME) / "task" / f"{split_name}.txt"
        with open(split_ids_file, "r") as f:
            return [line.strip() for line in f]
        
def _get_line_strings_from_json_file(filename: str) -> List[str]:
    """Get the text content of each line."""
    json_line_elements = _get_line_elements_from_json_file(filename)
    return [_get_text_from_json_element(el) for el in json_line_elements]

def _get_text_from_json_element(line_element: Dict[str, Any]) -> str:
    """Extract the text content from a JSON line element."""
    if "label" in line_element:
        return line_element["label"]
    return ""

def _get_line_elements_from_json_file(filename: str) -> List[Dict[str, Any]]:
    with open(filename, 'r', encoding='utf-8') as file:
        json_data = file.read()

        # Parse JSON data
        parsed_data = json.loads(json_data)

        # Extract the "line" elements
        lines = parsed_data["line"] if "line" in parsed_data else []

        return lines
        
if __name__ == "__main__":
    load_and_print_info(BIAM)
                 
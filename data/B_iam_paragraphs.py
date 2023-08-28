import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from data.base_data_module import BaseDataModule, load_and_print_info
from data.B_iam import BIAM
from data.data_util import BaseDataset, convert_strings_to_labels, resize_image
import metadata.b_iam_paragraphs as metadata
from stems.paragraph import ParagraphStem

IMAGE_SCALE_FACTOR = metadata.IMAGE_SCALE_FACTOR
MAX_LABEL_LENGTH = metadata.MAX_LABEL_LENGTH
NEW_LINE_TOKEN = metadata.NEW_LINE_TOKEN
PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME


class BIAMParagraphs(BaseDataModule):
    """IAM Handwriting database paragraphs."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "true").lower() == "true"

        self.mapping = metadata.MAPPING
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}

        self.input_dims = metadata.DIMS  # We assert that this is correct in setup()
        self.output_dims = metadata.OUTPUT_DIMS  # We assert that this is correct in setup()

        self.transform = ParagraphStem()
        self.trainval_transform = ParagraphStem(augment=self.augment)
    
    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser
    
    def prepare_data(self, *args, **kwargs) -> None:
        if (PROCESSED_DATA_DIRNAME / "_properties.json").exists():
            return

        iam = BIAM()
        iam.prepare_data()
        
        properties = {}
        for split in ["train", "val", "test"]:
            crops, labels = get_paragraph_crops_and_labels(iam=iam, split=split)
            save_crops_and_labels(crops=crops, labels=labels, split=split)

            properties.update(
                {
                    id_: {
                        "crop_shape": crops[id_].size[::-1],
                        "label_length": len(label),
                        "num_lines": _num_lines(label),
                    }
                    for id_, label in labels.items()
                }
            )
            
        with open(Path(PROCESSED_DATA_DIRNAME) / "_properties.json", "w") as f:
            json.dump(properties, f, indent=4)  
    
    def setup(self, stage: str = None) -> None:
        self.prepare_data()
        
        def _load_dataset(split: str, transform: Callable) -> BaseDataset:
            crops, labels = load_processed_crops_and_labels(split)
            Y = convert_strings_to_labels(strings=labels, mapping=self.inverse_mapping, length=self.output_dims[0])
            return BaseDataset(crops, Y, transform=transform)

        validate_input_and_output_dimensions(input_dims=self.input_dims, output_dims=self.output_dims)

        if stage == "fit" or stage is None:
            self.data_train = _load_dataset(split="train", transform=self.trainval_transform)
            self.data_val = _load_dataset(split="val", transform=self.transform)

        if stage == "test" or stage is None:
            self.data_test = _load_dataset(split="test", transform=self.transform)    
            
    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "B_IAM Paragraphs Dataset\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.input_dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        
        data = f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
        return basic + data         

def validate_input_and_output_dimensions(
    input_dims: Optional[Tuple[int, ...]], output_dims: Optional[Tuple[int, ...]]
) -> None:
    """Validate input and output dimensions against the properties of the dataset."""
    properties = get_dataset_properties()

    max_image_shape = properties["crop_shape"]["max"] / IMAGE_SCALE_FACTOR
    assert input_dims is not None and input_dims[1] >= max_image_shape[0] and input_dims[2] >= max_image_shape[1]

    # Add 2 because of start and end tokens
    assert output_dims is not None and output_dims[0] >= properties["label_length"]["max"] + 2

            
def get_paragraph_crops_and_labels(
    iam: BIAM, split: str, scale_factor=IMAGE_SCALE_FACTOR
) -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """Create BIAM paragraph crops and labels for a given split, with resizing."""
    crops = {}
    labels = {}
    for iam_id in iam.ids_by_split[split]:
        image = iam.load_image(iam_id)
        
        crops[iam_id] = image
        
        # Resize the cropped image
        crops[iam_id] = resize_image(crops[iam_id], scale_factor=scale_factor)
        
        # Get the paragraph string for the given BIAM form ID
        labels[iam_id] = iam.paragraph_string_by_id[iam_id]
        
    return crops, labels

def save_crops_and_labels(crops: Dict[str, Image.Image], labels: Dict[str, str], split: str):
    """Save crops, labels, and shapes of crops of a split for the BIAM dataset."""
    (Path(PROCESSED_DATA_DIRNAME) / split).mkdir(parents=True, exist_ok=True)

    with open(_labels_filename(split), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)

    for id_, crop in crops.items():
        crop.save(_crop_filename(id_, split))
        
def load_processed_crops_and_labels(split: str) -> Tuple[Sequence[Image.Image], Sequence[str]]:
    """Load processed crops and labels for the given split."""
    with open(_labels_filename(split), "r", encoding="utf-8") as f:
        labels = json.load(f)

    sorted_ids = sorted(labels.keys())
    ordered_crops = [Image.open(_crop_filename(id_, split)).convert("L") for id_ in sorted_ids]
    ordered_labels = [labels[id_] for id_ in sorted_ids]

    assert len(ordered_crops) == len(ordered_labels)
    return ordered_crops, ordered_labels 

def get_dataset_properties() -> dict:
    """Return properties describing the overall dataset."""
    with open(Path(PROCESSED_DATA_DIRNAME) / "_properties.json", "r", encoding="utf-8") as f:
        properties = json.load(f)

    def _get_property_values(key: str) -> list:
        return [_[key] for _ in properties.values()]

    crop_shapes = np.array(_get_property_values("crop_shape"))
    aspect_ratios = crop_shapes[:, 1] / crop_shapes[:, 0]
    return {
        "label_length": {
            "min": min(_get_property_values("label_length")),
            "max": max(_get_property_values("label_length")),
        },
        "num_lines": {"min": min(_get_property_values("num_lines")), "max": max(_get_property_values("num_lines"))},
        "crop_shape": {"min": crop_shapes.min(axis=0), "max": crop_shapes.max(axis=0)},
        "aspect_ratio": {"min": aspect_ratios.min(), "max": aspect_ratios.max()},
    }     
        
def _labels_filename(split: str) -> Path:
    """Return filename of processed labels."""
    return Path(PROCESSED_DATA_DIRNAME) / split / "_labels.json"        

def _crop_filename(id_: str, split: str) -> Path:
    """Return filename of processed crop."""
    return Path(PROCESSED_DATA_DIRNAME) / split / f"{id_}.png"

def _num_lines(label: str) -> int:
    """Return number of lines of text in label."""
    return label.count(NEW_LINE_TOKEN) + 1

if __name__ == "__main__":
    load_and_print_info(BIAMParagraphs)

         

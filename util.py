"""Utility functions"""
import os
import contextlib
from pathlib import Path
from typing import Union


import numpy as np
from PIL import Image
import smart_open
from tqdm import tqdm


def to_categorical(y, num_classes):
    """1-hot encode a tensor."""
    return np.eye(num_classes, dtype="uint8")[y]


def read_image_pil(image_uri: Union[Path, str], grayscale=False) -> Image:
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_image_pil_file(image_file, grayscale=False) -> Image:
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
        return image
    
@contextlib.contextmanager
def temporary_working_directory(working_dir: Union[str, Path]):
    """Temporarily switches to a directory, then returns to the original directory on exit."""
    curdir = os.getcwd()
    os.chdir(working_dir)
    try:
        yield
    finally:
        os.chdir(curdir)    



class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)
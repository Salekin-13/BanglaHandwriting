"""Base Dataset class."""
from typing import Any, Dict, List, Callable, Sequence, Tuple, Union

from PIL import Image
import torch
import re

SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset class that simply processes data and targets through optional transforms.

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """
        datum, target = self.data[index], self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target

def convert_strings_to_labels(strings: List[str], mapping: Dict[str, int], length: int) -> torch.Tensor:
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]
    for i, string in enumerate(strings):
        string = string.replace("&garb", "<G>")
        string = string.replace("&under", "<U>")
        string = string.replace("&eng", "<ENG>")
        string = string.replace("&deg", "<DEG>")
        
        tokens = re.findall(r"(\s+|<G>|<U>|<ENG>|<DEG>|[^\s<GUE>]+)", string)  # Split the modified string
        
        char_tokens = []
        for token in tokens:
            if token in ["<G>", "<U>", "<ENG>", "<DEG>"]:
                char_tokens.append(token)
            else:
                char_tokens.extend(list(token))  # Split the word into characters
                
        tokens = ["<S>", *char_tokens,"<E>"]  # Add the special start token
        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping.get(token, mapping["<P>"])  # Use mapping["<P>"] if token not in mapping
    return labels

def split_dataset(base_dataset: BaseDataset, fraction: float, seed: int) -> Tuple[BaseDataset, BaseDataset]:
    """
    Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
    other of size (1 - fraction) * size of the base_dataset.
    """
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size
    return torch.utils.data.random_split(  # type: ignore
        base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    )


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((image.width // scale_factor, image.height // scale_factor), resample=Image.BILINEAR)
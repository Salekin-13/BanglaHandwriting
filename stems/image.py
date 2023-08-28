import torch
from torchvision import transforms


class ImageStem:
    def __init__(self):
        self.pil_transforms = transforms.Compose([])
        self.pil_to_tensor = transforms.ToTensor()
        self.torch_transforms = torch.nn.Sequential()

    def __call__(self, img):
        img = self.pil_transforms(img)
        img = self.pil_to_tensor(img)

        with torch.no_grad():
            img = self.torch_transforms(img)

        return img
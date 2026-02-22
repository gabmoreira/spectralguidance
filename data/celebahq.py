import os
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Tuple

def celeba_transform(augment: bool):
    if augment:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    return transform

class CelebAHQMask(Dataset):
    def __init__(self, split: str, augment: bool, cache_dir: str):
        self.cache_dir = cache_dir
        self.root = "CelebAMask-HQ"
        self.attr_filename = "CelebAMask-HQ-attribute-anno.txt"
        attr_file = os.path.join(cache_dir, self.root, self.attr_filename)
        
        with open(attr_file, 'r') as f:
            lines = f.readlines()
        
        self.classes = lines[1].strip().split()
        self.filenames = []
        self.attributes = []
        data_lines = lines[2:]
        for line in data_lines:
            parts = line.strip().split()
            self.filenames.append(parts[0])
            self.attributes.append([int(x) for x in parts[1:]])
        
        self.transform = celeba_transform(augment)

    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        filename = self.filenames[index]
        img_path = os.path.join(self.cache_dir, self.root, "CelebA-HQ-img256", filename)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        attrs = (torch.tensor(self.attributes[index]) > 0).long()
        return img, attrs

    def collate_fn(self, batch) -> Tuple[Tensor, Tensor]:
        imgs = torch.stack([item[0] for item in batch], dim=0)     # (B, 3, 256, 256)
        attrs = torch.stack([item[1] for item in batch], dim=0)    # (B, 40)
        return imgs, attrs
    
    def get_guidance(self, target: int) -> Tensor:
        one_hot = torch.tensor(self.attributes) > 0
        p_y_x0 = one_hot[:,target].float()
        return p_y_x0
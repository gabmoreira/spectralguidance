import os
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Tuple

def preload_masks(mask_root):
    all_masks = {}
    for folder_name in [f'{i}' for i in range(15)]:
        folder_path = os.path.join(mask_root, folder_name)
        if not os.path.exists(folder_path):
            continue

        for filename in tqdm(os.listdir(folder_path), desc=f"Loading folder {folder_name}"):
            if not filename.endswith('.png'):
                continue
            
            img_str, attribute_name = filename.split('_', 1)
            attribute_name = attribute_name.replace('.png', '')
            image_number = int(img_str)
            
            mask_path = os.path.join(folder_path, filename)
            if attribute_name != "hair":
                continue
            with Image.open(mask_path) as im:
                im = im.resize((256, 256), resample=Image.NEAREST)
                all_masks[image_number] = np.array(im, dtype=bool)
    
    return all_masks

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
        if self.transform is not None:
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
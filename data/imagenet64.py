import os
import json
import pickle
import numpy as np
import logging
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def imagenet_transform(augment: bool):
    if augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),                
            transforms.Normalize(                
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),                
            transforms.Normalize(                
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]
            )
        ])
    return transform

def load_imagenet_label_map(file_path):
    """
    Creates a mapping from the 0-indexed targets 
    to human-readable strings.
    """
    idx_to_name = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            # The file format is: WNID NumericID HumanName
            # Example: n02119789 1 kit_fox
            parts = line.strip().split()
            
            wnid = parts[0]
            numeric_id = int(parts[1])
            human_name = " ".join(parts[2:]).replace('_', ' ')
            
            idx_to_name[numeric_id - 1] = human_name
            
    return idx_to_name
    

class ImageNet64(Dataset):
    def __init__(self, split: str, augment: bool, cache_dir: str, class_list: list[int] | None = None):
        self.transform = imagenet_transform(augment)
        
        base_dir = f"{cache_dir}/imagenet64"
        self.label_map = load_imagenet_label_map(f'{base_dir}/map_clsloc.txt')

        self.data, self.targets = [], []
        for i in range(1, 11):
            logger.info(f"Loading {base_dir}/train_data_batch_{i}")
            with open(f"{base_dir}/train_data_batch_{i}", 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend([label - 1 for label in entry['labels']])
        self.data = np.vstack(self.data).reshape(-1, 3, 64, 64).transpose((0, 2, 3, 1))

        self.class_to_indices = json.load(open(f"{base_dir}/class_index.json"))
        self.class_to_indices = {int(k): v for k, v in self.class_to_indices.items()}

        if class_list is not None:
            logger.info("Filtering classes")
            class_set = set(class_list)
            keep = np.array([i for i, t in enumerate(self.targets) if t in class_set])
            self.data = self.data[keep]
            self.targets = [self.targets[i] for i in keep]
            self.class_to_indices = {}
            for new_idx, t in enumerate(self.targets):
                self.class_to_indices.setdefault(t, []).append(new_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

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

class ImageNet64(Dataset):
    def __init__(self, split: str, augment: bool, cache_dir: str):
        self.data = []
        self.targets = []
        self.transform = imagenet_transform(augment)

        file_list = [cache_dir + f"/imagenet64/train_data_batch_{i}" for i in range(1,11)]

        for file_path in file_list:
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                # Labels in ImageNet 32/64 are often 1-indexed; 
                # subtract 1 to make them 0-indexed for PyTorch
                self.targets.extend([label - 1 for label in entry['labels']])

        self.data = np.vstack(self.data).reshape(-1, 3, 64, 64)
        # Transpose to (N, H, W, C) so PIL can read it easily before transforms
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
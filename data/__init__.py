from torchvision import transforms, datasets

from .celebahq import CelebAHQMask
from .imagenet64 import ImageNet64, load_imagenet_label_map, imagenet_transform
from .cifar10 import CIFAR10
    
def get_dataset(dataset: str, split: str, augment: bool, cache_dir: str):
    if dataset == "imagenet64":
        return ImageNet64(split, augment, cache_dir)
    
    elif dataset == "celeba-hq-mask":
        return CelebAHQMask(split, augment, cache_dir)
    
    elif dataset == "mnist":
        if augment:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.ToTensor()

        return datasets.MNIST(
            root=cache_dir,
            train=split == "train",
            download=True,
            transform=transform
        )
    
    elif dataset == "cifar10":
        if augment:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        return CIFAR10(
            root=cache_dir,
            train=split == "train",
            download=True,
            transform=transform
        )
    
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'.")
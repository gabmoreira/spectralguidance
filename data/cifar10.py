import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets

class CIFAR10(datasets.CIFAR10):
    def get_guidance(self, target: int) -> Tensor:
        labels = torch.tensor(self.targets, dtype=torch.long)
        one_hot = F.one_hot(labels, num_classes=10)
        p_y_x0 = one_hot[:, target].float()
        return p_y_x0
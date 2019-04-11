import torch
import torch.nn.functional as F


def binary_cross_entropy(output, target):
    return F.binary_cross_entropy(output, target)
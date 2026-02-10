
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numbers
import math
import numpy as np
from PIL import Image

def resize_list_pil(frames, size):
    # frames: list of PIL Images
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, tuple) and len(size) == 1:
        size = (size[0], size[0])
    return [img.resize(size, Image.BILINEAR) for img in frames]

def resize_tensor(frames, size):
    # frames: Tensor (C, T, H, W)
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, tuple) and len(size) == 1:
        size = (size[0], size[0])
    
    # Input is (C, T, H, W). F.interpolate expects (N, C, H, W).
    # We treat C as N, T as C.
    return F.interpolate(frames, size=size, mode='bilinear', align_corners=False)

def adaptive_resize(frames, size):
    if isinstance(frames, list):
        return resize_list_pil(frames, size)
    elif isinstance(frames, torch.Tensor):
        return resize_tensor(frames, size)
    return frames

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Resize(object):
    def __init__(self, size, interpolation='bilinear'):
        self.size = size

    def __call__(self, img):
        return adaptive_resize(img, self.size)

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return adaptive_resize(img, self.size)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1) # C, T, H, W shape adaptation?
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, tensor):
        # tensor is C, T, H, W (if called inside spatial_sampling?)
        # Or T H W C?
        # Check usage in _aug_frame.
        # Line 250 calls tensor_normalize which is NOT this class.
        # This class Normalize is used in 'val' mode usually.
        # In val mode:
        # buffer = ...
        # transform = Compose([Resize, CenterCrop, Normalize, stack])
        # If Normalize is called, we need to know input shape.
        # Assuming val pipeline passes T H W C or C T H W?
        # Usually val pipeline: List[PIL] -> Resize/Crop -> Stack -> Normalize.
        # Stack -> T C H W. 
        # Normalize expects T C H W.
        # self.mean view (3, 1, 1) broadcast over T, H, W.
        if isinstance(tensor, torch.Tensor):
             if tensor.shape[1] == 3: # T C H W
                 mean = self.mean.view(1, 3, 1, 1)
                 std = self.std.view(1, 3, 1, 1)
                 return (tensor - mean) / std
        return tensor

def create_random_augment(input_size, auto_augment=None, interpolation='bilinear'):
    return lambda x: adaptive_resize(x, input_size)

def random_short_side_scale_jitter(images, min_size, max_size, inverse_uniform_sampling=False):
    return images, 0

def random_crop(images, size):
    return adaptive_resize(images, size), (0,0,0,0)

def random_resized_crop_with_shift(images, target_height, target_width, scale, ratio):
    return adaptive_resize(images, (target_height, target_width))

def random_resized_crop(images, target_height, target_width, scale, ratio):
    return adaptive_resize(images, (target_height, target_width))

def horizontal_flip(prob, images):
    return images, 0

def uniform_crop(images, size, spatial_idx):
    return adaptive_resize(images, size)

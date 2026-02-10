
import torch

class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """
    def __init__(self, div_255=True):
        self.div_255 = div_255

    def __call__(self, clip):
        if isinstance(clip[0], torch.Tensor):
            return torch.stack(clip)
        
        # Assume clip is list of PIL images
        import torchvision.transforms.transforms as t
        return torch.stack([t.ToTensor()(img) for img in clip])

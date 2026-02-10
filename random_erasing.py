
class RandomErasing(object):
    def __init__(self, probability=0.5, mode='pixel', max_count=1, num_splits=0, device='cpu'):
        self.probability = probability

    def __call__(self, img):
        return img

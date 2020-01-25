# Taken from https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/utils.py

import numpy as np

def split_train_and_validation(dataset, random_seed=42, validation_split=0.2, shuffle=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, validation_indices = indices[split:], indices[:split]
    return train_indices, validation_indices


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, weight=1.0):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def value(self):
        return self.val

    def average(self):
        return self.avg
            
    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count
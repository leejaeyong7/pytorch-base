# -*- coding: utf-8 -*-
"""Boilerplate Dataset Class definition.

Defines example dataset that can be used as a base for actual dataset.

Model extends torch.utils.data.Dataset which should override __init__, __len__
and __getitem__ functions.

This class will be then used by DataLoader class that handles sampling / 
batching, etc.
"""

class BarDataset(Dataset):
    def __init__(self, dataset_dir):
        self.samples = []
        # TODO: add samples into self.samples somehow

    def __len__(self):
        """ Returns total number of samples in this dataset. """
        return len(self.samples)

    def __getitem__(self, index):
        """ Gets sample in this dataset by index. """
        return self.samples[index]

"""
This module implements an interface for all dataloaders.
"""
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import dataloaders.collate_functions

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size=32, num_workers=8,
        collate_fn='mix_collate_fn', pin_memory=True, drop_last=True):

        self.collate_fn = getattr(dataloaders.collate_functions, collate_fn)

        self.batch_idx = 0

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'collate_fn': self.collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last
        }

        # set shuffle to false, it is mutually exclusive with the sampler (?)
        self.shuffle = False
        idx_full = np.arange(len(dataset))
        self.train_sampler = SubsetRandomSampler(idx_full)

        super().__init__(sampler=self.train_sampler, **self.init_kwargs)
        #print(f'drop last: {self.drop_last}')

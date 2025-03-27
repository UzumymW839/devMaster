"""
This module implements an abstract base class for inheritance.
All datasets should inherit from it for testing compatibility.

Datasets are initialized with a file `file_list_filename` in csv format which contains all file names
to be included in the dataset.
"""

from abc import ABC, abstractmethod
import torch
import pandas as pd


class BaseDataset(torch.utils.data.Dataset, ABC):
    """
    This is an abstract base class for datasets.
    """

    def __init__(self, file_list_filename, configuration={}) -> None:
        """
        Initialize the class.
        """
        super().__init__()
        self.filenames_dataframe = pd.read_csv(file_list_filename)
        self.configuration = configuration


    def __len__(self) -> int:
        """
        Return the total number of images in the dataset.
        """
        return len(self.filenames_dataframe)


    @abstractmethod
    def __getitem__(self, index: int):
        """Return a data point (usually data and labels in
            a supervised setting).
        """
        raise NotImplementedError


    @abstractmethod
    def get_full_example(self, index: int):
        """
        Read a full example from the dataset.
        """
        raise NotImplementedError


    def get_random_example(self):
        """
        Read a randomly chosen full example from the dataset.
        """
        random_index = torch.randint(low=0, high=len(self.noise_list), size=[1])
        return self.get_full_example(index=random_index)

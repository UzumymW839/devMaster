"""
This module defines the interface for models.
All models in this repository should inherit from it and implement the abstract methods.
"""

from abc import ABC, abstractmethod
import torch


class BaseModel(torch.nn.Module, ABC):
    """
    This is an abstract base class for pytorch models.
    """

    def __init__(self, configuration=None) -> None:
        """
        Initialize the class.

        Parameters:
            configuration:  Configuration dictionary.
        """
        super().__init__()

        self.configuration = configuration
        self.is_in_finetune_mode = False


    @abstractmethod
    def forward(self, input_data):
        """
        Forward the input data through the model and compute the output.

        Parameters:
            input_data: the input data to process.
        """
        raise NotImplementedError


    @abstractmethod
    def flatten_all_params(self):
        raise NotImplementedError


    """
    @abstractmethod
    def set_to_finetune_mode(self, path_pretrained_dict):
        raise NotImplementedError
    """

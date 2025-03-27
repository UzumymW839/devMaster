from abc import abstractmethod
import pathlib
import random
import time
import torch
import numpy as np
from utils.training_utils import save_maxscaled_audio


class BaseTester:
    """
    Base class for all testers
    """
    def __init__(self, model, metric_functions, device, dataloader, config,
                 test_output_folder: pathlib.Path, results_filename='results.npy'):

        self.test_output_folder = test_output_folder
        #self.test_output_folder.mkdir(exist_ok=False)
        self.test_output_folder.mkdir(exist_ok=True)
        self.results_filename = self.test_output_folder.joinpath(results_filename)
        print(self.results_filename)

        self.config = config

        self.model = model
        self.metric_functions = metric_functions

        self.device = device
        self.dataloader = dataloader


    @abstractmethod
    def _test_on_dataset(self):
        raise NotImplementedError


    def test(self):
        """
        Full training logic
        """
        result = self._test_on_dataset()

        np.save(self.results_filename, result)


    def export_example(self, output: torch.Tensor, target: torch.Tensor, data: torch.Tensor, snr_label: str):

        with torch.no_grad():

            inp = data.detach().cpu() # select target reference channel
            tar = target.detach().cpu() # target is single-channel anyway
            out = output.detach().cpu() # target is single-channel anyway

            # save single- and multi-channel audio example
            save_maxscaled_audio(inp, audio_name_tag='noisy input'+snr_label, audio_example_path=self.audio_example_path)
            save_maxscaled_audio(tar, audio_name_tag='target'+snr_label, audio_example_path=self.audio_example_path)
            save_maxscaled_audio(out, audio_name_tag='output'+snr_label, audio_example_path=self.audio_example_path)


    def compute_all_metrics(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """
        compute metrics for each example and then average.
        Parameters:
            output: Tensor containing the output of the model.
            target: the desired output as tensor.
        """
        batch_results = []
        for met in self.metric_functions:
            result_list = []
            for out, targ in zip(output, target): # zip over tensors makes the list items be (1, ...)

                #t0 = time.time()
                result_list.append(met(out, targ))
                #print(f'{met.__name__}: {time.time()-t0}')

            result = torch.nanmean(torch.tensor(result_list))
            batch_results.append(result)

        return batch_results


    def use_fixed_seed(self):
        seed = 123
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

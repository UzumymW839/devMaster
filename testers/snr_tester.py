from itertools import islice
import pathlib
import torch
import numpy as np
from tqdm import tqdm

from testers.base_tester import BaseTester


class SNRTester(BaseTester):

    def __init__(self, SNR_list, **kwargs):
        super().__init__(**kwargs)
        self.SNR_list = SNR_list

        # epoch-based test
        self.len_epoch = len(self.dataloader)


        self.audio_example_path = pathlib.Path(self.test_output_folder).joinpath('best_audio_examples')
        self.audio_example_path.mkdir(exist_ok=True)

        # split for speedup over noise types -> for each SNR, all speech files are used once
        self.batches_per_split = int(np.floor(len(self.dataloader) / len(self.SNR_list)))
        print(len(self.dataloader))
        print(self.batches_per_split)


    def _test_on_dataset(self):
        """
        Test the entire test dataset

        :return: A log that contains information about test results
        """
        self.use_fixed_seed()
        self.model.eval()

        # create result array
        results = np.zeros((self.batches_per_split, len(self.SNR_list), len(self.metric_functions)))

        with torch.no_grad():
            for snr_idx, snr_db in enumerate(self.SNR_list):
                print(f"now starting to process SNR of {snr_db} dB! ...")

                # set new SNR
                self.dataloader.dataset.noise_transform.set_fixed_snr(snr_db)

                # for each snr, only use a fraction of the data
                g_start = snr_idx * self.batches_per_split
                g_stop = g_start + self.batches_per_split
                this_generator = islice(self.dataloader, g_start, g_stop)

                for batch_idx, (data, target) in enumerate(tqdm(this_generator)):

                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(data)

                    # compute metrics on the complete batch
                    batch_result = self.compute_all_metrics(output=output, target=target)

                    results[batch_idx, snr_idx, :] = batch_result

                    if batch_idx == 0:
                        # save audio example
                        self.export_example(output=output[-1, ...], target=target[-1, ...], data=data[-1,...], snr_label=f'_{snr_db}dB_SNR')

                print(f'results for snr={snr_db}dB:')
                print(self.metric_functions)
                print(np.nanmedian(results[:, snr_idx, :], axis=0))

        return results


    def _progress(self, batch_idx: int) -> str:
        """
        Format a string indicating the progress in the current epoch.
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.dataloader, 'n_samples'):
            current = batch_idx * self.dataloader.batch_size
            total = self.dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

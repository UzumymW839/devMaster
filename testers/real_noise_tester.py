import pathlib
import torch
import numpy as np
from tqdm import tqdm
from testers.base_tester import BaseTester

class RealNoiseTester(BaseTester):
    def __init__(self, noise_type_list, SNR_list, **kwargs):
        super().__init__(**kwargs)
        self.SNR_list = SNR_list

        self.noise_type_list = noise_type_list

        self.audio_example_path = pathlib.Path(self.test_output_folder).joinpath('best_audio_examples_real_noise')
        self.audio_example_path.mkdir(exist_ok=True)

        self.results_filename = self.test_output_folder.joinpath('results_real_noise.npy')


    def _test_on_dataset(self):
        """
        Test the entire test dataset

        :return: A log that contains information about test results
        """
        self.use_fixed_seed()
        self.model.eval()

        # create result array

        results = np.zeros((len(self.dataloader), len(self.noise_type_list), len(self.metric_functions), len(self.SNR_list)))

        with torch.no_grad():

            # set new SNR
            for snr_idx, fixed_snr_db in enumerate(self.SNR_list):
                print(f"now starting to process at {fixed_snr_db} dB SNR! ...")
                self.dataloader.dataset.noise_transform.set_fixed_snr(fixed_snr_db)

                for noise_idx, noise_type_str in enumerate(self.noise_type_list):
                    print(f"now starting to process {noise_type_str} noise! ...")
                    self.dataloader.dataset.noise_transform.set_noise_type(noise_type_str)

                    for batch_idx, (data, target) in enumerate(tqdm(self.dataloader)):

                        data, target = data.to(self.device), target.to(self.device)

                        output = self.model(data)

                        # compute metrics on the complete batch
                        batch_result = self.compute_all_metrics(output=output, target=target)

                        results[batch_idx, noise_idx, :, snr_idx] = batch_result

                        if batch_idx == 0:
                            # save audio example
                            self.export_example(output=output[-1, ...], target=target[-1, ...], data=data[-1,...],
                                                snr_label=f'_{noise_type_str}_{fixed_snr_db}dB_SNR')

                    print(f'results for noise = {noise_type_str} at {fixed_snr_db} dB:')

                    print(self.metric_functions)
                    print(np.nanmedian(results[:, noise_idx, :, snr_idx], axis=0))

                print(f"means over noise types for SNR={fixed_snr_db} dB: ")
                print(np.mean(np.nanmedian(results[:, :, :, snr_idx], axis=0), axis=0))

        return results




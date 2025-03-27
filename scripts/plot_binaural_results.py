import pathlib
import numpy as np
import matplotlib.pyplot as plt

import metrics
from utils.json_handling import read_json

cmap = plt.get_cmap('tab10')

RESULTS_PATH = pathlib.Path('saved/test_output')
MODELS_PATH = pathlib.Path('saved/models')

out_folder = RESULTS_PATH.joinpath('plots')

dict_list = [
    {"configuration_name": 'noisy_binaural_oem_dummy', "run_string": 'default_run', "label": 'Noisy OEMs', 'color': cmap(0), 'linestyle': '-'},
    {"configuration_name": 'binaural_experiment', "run_string": '0830_091644', "label": 'bDNN (reverb.)', 'color': cmap(2), 'linestyle': '-'},

]

test_config = read_json('test_configs/binaural_snr_test.json')

plot_string = 'simulated_data_binaural_snr_results_'

result_list = []

for dnn_dict in dict_list:
    configuration_name = dnn_dict["configuration_name"]
    print(configuration_name)
    run_string = dnn_dict["run_string"]

    run_result_folder = RESULTS_PATH.joinpath(configuration_name, run_string)
    results_fname = run_result_folder.joinpath('results.npy')

    # array has dimensions:
    #   (len(self.dataloader), len(self.metric_functions), len(self.SNR_list))
    results_array = np.load(results_fname, allow_pickle=True)
    print(results_array.shape)

    # Read metrics and SNRs corresponding to the results array
    metric_functions = [getattr(metrics, met)() for met in test_config['metrics']]
    SNR_list = test_config['tester']['args']['SNR_list']

    #print(results_array.shape)
    result_list.append(np.median(results_array, axis=0)) # median over batches

all_results = np.stack(result_list, axis=2) # shape: (snrs, metrics, algorithms)
print(all_results.shape)

for n_metric, met in enumerate(metric_functions):

    plot_fname = out_folder.joinpath(plot_string+met.__name__+'.png')

    plt.figure(figsize=(8, 6))
    plt.ylabel(met.__name__)
    plt.xlabel('SNR / dB')
    plt.xticks(ticks=np.arange(len(SNR_list)), labels=SNR_list)
    plt.grid()

    for n_dnn, dnn_dict in enumerate(dict_list):
        plt.plot(
            all_results[:, n_metric, n_dnn],
            label=dnn_dict['label'], linestyle=dnn_dict['linestyle'],
            marker='o', color=dnn_dict['color'])

    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(plot_fname)



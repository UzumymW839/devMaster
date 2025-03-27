import pathlib
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import metrics
from utils.json_handling import read_json

cmap = plt.get_cmap('tab10')

RESULTS_PATH = pathlib.Path('saved/test_output')
MODELS_PATH = pathlib.Path('saved/models')

out_folder = RESULTS_PATH.joinpath('plots')

dict_list = [
    {"configuration_name": 'noisy_oem_dummy', "run_string": 'default_run', "label": 'keine Verarbeitung', 'color': cmap(0), 'linestyle': '-'},
    #{"configuration_name": 'diamond_5mic_tiny', "run_string": '0524_155539', "label": 'FT_JNF Tiny', 'color': cmap(1), 'linestyle': '-'}, # half model without KD
    #{"configuration_name": 'diamond_5mic_vtiny', "run_string": '0526_175613', "label": 'FT_JNF vTiny', 'color': cmap(2), 'linestyle': '-'}, # quarter model without KD
    #{"configuration_name": 'diamond_5mic_experiment', "run_string": '0502_161633', "label": 'FT_JNF', 'color': cmap(3), 'linestyle': '-'}, # teacher model
    {"configuration_name": 'diamond_5mic_experiment', "run_string": '0801_143408', "label": 'Lehrer', 'color': cmap(1), 'linestyle': '-'}, # teacher with 8 batch size
    #{"configuration_name": 'diamond_TwoSteps', "run_string": '0610_135810', "label": 'tiny FT_JNF KD MAE', 'color': cmap(6), 'linestyle': '-'}, # half model with elementwise absolute error with KD
    #{"configuration_name": 'diamond_TwoSteps', "run_string": '0614_233313', "label": 'vtiny FT_JNF KD MAE', 'color': cmap(7), 'linestyle': '-'}, # quarter model with elementwise absolute error with KD
    {"configuration_name": "diamond_normal_trainlist", "run_string": '0917_105029', "label": 'Schüler ohne KD', 'color': cmap(2), 'linestyle': '-'}, # target model with normal training 64/16er: 0624_120334 5_mic_target
    {"configuration_name": 'diamond_TwoSteps', "run_string": '0808_155225', "label": 'Schüler KD Mask', 'color': cmap(4), 'linestyle': '-'}, # target model with Mask KD MAE
    {"configuration_name": 'diamond_TwoSteps', "run_string": '0701_172332', "label": 'Schüler KD LSTM 1', 'color': cmap(7), 'linestyle': '-'}, # target model with Regressor KD
    {"configuration_name": 'diamond_LinD', "run_string": '0828_232658', "label": 'Schüler KD Lin', 'color': cmap(5), 'linestyle': '-'}, # target model MAE between Linear Layers 64/15er: 0806_165806 diamond_twoSteps
    {"configuration_name": 'diamond_TS_ALLD', "run_string": '0817_200225', "label": 'Schüler KD LSTM 2', 'color': cmap(3), 'linestyle': '-'},
    #{"configuration_name": 'diamond_TwoSteps', "run_string": '1023_132130', "label": 'Schüler LSTM 1', 'color': cmap(7), 'linestyle': '-'}, # between first LSTM layer
    {"configuration_name": 'diamond_TwoSteps', "run_string": '1001_173125', "label": 'Schüler KD LSTM 1 + 2 + Lin', 'color': cmap(6), 'linestyle': '-'}, # t model MSE between Linear Layers
    #{"configuration_name": 'diamond_TwoSteps', "run_string": '0721_140838', "label": 't FT_JNF KD MSE-All', "color": cmap(7), "linestyle": '-'} # t model MSE between all layers
]

test_config = read_json('test_configs/diamond_snr_test.json')

plot_string = 'simulated_data_snr_results_' # UMBENNEN!

result_list = []
yerr = []

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
    yerr.append(np.var(results_array, axis=0)) # variance over batches

all_results = np.stack(result_list, axis=2) # shape: (snrs, metrics, algorithms)
all_yerr = np.stack(yerr, axis=2)
print(all_results.shape)

for n_metric, met in enumerate(metric_functions):

    plot_fname = out_folder.joinpath(plot_string+met.__name__+'.png')

    plt.figure(figsize=(8, 6))
    plt.ylabel(met.__name__)
    plt.xlabel('SNR / dB')
    plt.xticks(ticks=np.arange(len(SNR_list)), labels=SNR_list)
    plt.grid()

    #for n_dnn, dnn_dict in enumerate(dict_list):
    #    plt.plot(
    #        all_results[:, n_metric, n_dnn],
    #        label=dnn_dict['label'], linestyle=dnn_dict['linestyle'],
    #        marker='o', color=dnn_dict['color'])
    for n_dnn, dnn_dict in enumerate(dict_list):
        plt.errorbar(
            np.arange(len(SNR_list)),
            all_results[:, n_metric, n_dnn],
            yerr=all_yerr[:, n_metric, n_dnn],
            label=dnn_dict['label'], linestyle=dnn_dict['linestyle'],
            marker='o', color=dnn_dict['color'], capsize=5
        )

    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(plot_fname)



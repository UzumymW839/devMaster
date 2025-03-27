import pathlib
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import os

import metrics
from utils.json_handling import read_json

cmap = plt.get_cmap('tab10')

# get the parent folder
parent_folder = pathlib.Path(__file__).parent.parent

RESULTS_PATH = pathlib.Path(os.path.join(parent_folder, 'saved','test_output'))
MODELS_PATH = pathlib.Path(os.path.join(parent_folder, 'saved','models'))

out_folder = RESULTS_PATH.joinpath('plots')

dict_list = [
    #{"configuration_name": 'noisy_oem_dummy', "run_string": 'default_run', "label": 'Noisy OEM', 'color': cmap(0), 'linestyle': '-'},
    {"configuration_name": 'diamond_OneShot', "run_string": '0906_084145', "label": 'Ein-Schritt', "MACs": '1116358656', "color": cmap(1), "linestyle":'-'}, # Lin: 80/32
    {"configuration_name": 'diamond_LinD', "run_string": '0828_232658', "label": 'Zwei-Schritt', "MACs": '1116358656', "color": cmap(2), "linestyle":'-'}, # LinD: 80/32
    {"configuration_name": 'diamond_ThreeShot', "run_string": '0911_111425', "label": 'Drei-Schritt', "MACs": '1116358656', "color": cmap(3), "linestyle":'-'}, # LinD: 80/32
]

#test_config = read_json(parent_folder + 'test_configs/ALLD_test.json')
test_config = read_json(os.path.join(parent_folder, 'test_configs', 'ALLD_test.json'))

plot_string = 'OneTwoThree_' # UMBENNEN!

result_list = []
yerr = []

for dnn_dict in dict_list:
    configuration_name = dnn_dict["configuration_name"]
    print(configuration_name)
    run_string = dnn_dict["run_string"]

    run_result_folder = RESULTS_PATH.joinpath(configuration_name, run_string)
    results_fname = run_result_folder.joinpath('results.npy')

    results_array = np.load(results_fname, allow_pickle=True)
    print(results_array.shape)

    metric_functions = [getattr(metrics, met)() for met in test_config['metrics']]
    SNR_list = test_config['tester']['args']['SNR_list']

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

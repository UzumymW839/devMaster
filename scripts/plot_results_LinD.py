import pathlib
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
# select the parent folder
parentfolder = str(pathlib.Path(__file__).parent.parent)

#import metrics
from utils.json_handling import read_json

import pandas as pd

cmap = plt.get_cmap('tab10')

#parent folder + 'saved/test_output')
RESULTS_PATH = pathlib.Path(parentfolder + '/saved/test_output')
MODELS_PATH = pathlib.Path(parentfolder + 'saved/models')

out_folder = RESULTS_PATH.joinpath('plots')

experiment = pd.DataFrame()

dict_list1 = [
    #{"configuration_name": 'noisy_oem_dummy', "run_string": 'default_run', "label": 'Noisy OEM', 'color': cmap(0), 'linestyle': '-'},
    #{"configuration_name": 'diamond_5mic_experiment', "run_string": '0801_143408', "label": 'FT_JNF', 'color': cmap(1), 'linestyle': '-'}, # teacher with 8 batch size
    {"configuration_name": 'diamond_LinD', "run_string": '0905_162426', "label": '256/64', "color": cmap(5), "linestyle":'-'}, # LinD: 256/64
    {"configuration_name": 'diamond_LinD', "run_string": '0903_173913', "label": '128/32', "color": cmap(5), "linestyle":'-'}, # LinD: 128/32
    {"configuration_name": 'diamond_LinD', "run_string": '0903_173255', "label": '88/40', "color": cmap(5), "linestyle":'-'}, # LinD: 88/40
    {"configuration_name": 'diamond_LinD', "run_string": '0828_232658', "label": '80/32', "color": cmap(5), "linestyle":'-'}, # LinD: 80/32
    {"configuration_name": 'diamond_LinD', "run_string": '0828_232323', "label": '72/24', "color": cmap(5), "linestyle":'-'}, # LinD: 72/24
    {"configuration_name": 'diamond_LinD', "run_string": '0822_103010', "label": '64/16', 'color': cmap(5), 'linestyle': '-'}, # target model MAE between lin layer b:0805_103641
    {"configuration_name": 'diamond_LinD', "run_string": '0824_181818', "label": '56/8', "color": cmap(5), "linestyle":'-'}, # LinD: 56/8
    {"configuration_name": 'diamond_LinD', "run_string": '0826_110300', "label": '48/8',  "color": cmap(5), "linestyle":'-'}, # LinD: 48/8
]
dict_list2 = [
    {"configuration_name": 'diamond_TwoSteps', "run_string": '1004_084719', "label": '256/64', 'color': cmap(0), 'linestyle': '-'},
    {"configuration_name": 'diamond_TwoSteps', "run_string": '1003_110545', "label": '128/32', 'color': cmap(1), 'linestyle': '-'},
    {"configuration_name": 'diamond_TwoSteps', "run_string": '1001_173734', "label": '88/40', 'color': cmap(7), 'linestyle': '-'},
    {"configuration_name": 'diamond_TwoSteps', "run_string": '1001_173125', "label": '80/32', 'color': cmap(5), 'linestyle': '-'},
    {"configuration_name": 'diamond_TwoSteps', "run_string": '1004_112442', "label": '72/24', 'color': cmap(8), 'linestyle': '-'},
    {"configuration_name": 'diamond_TwoSteps', "run_string": '0927_171938', "label": '64/16', 'color': cmap(6), 'linestyle': '-'}, # t model MSE between all layers
    {"configuration_name": 'diamond_TwoSteps', "run_string": '1007_135345', "label": '56/8', 'color': cmap(3), 'linestyle': '-'},
    {"configuration_name": 'diamond_TwoSteps', "run_string": '1009_162003', "label": '48/8', 'color': cmap(4), 'linestyle': '-'},
]
dict_list3 = [
    {"configuration_name": 'diamond_normal_trainlist', "run_string": '0913_115746', "label": '256/64', "color": cmap(1), "linestyle":'-'}, # NT: 256/64
    {"configuration_name": 'diamond_normal_trainlist', "run_string": '0916_111601', "label": '128/32', "color": cmap(2), "linestyle":'-'}, # NT: 128/32
    {"configuration_name": 'diamond_normal_trainlist', "run_string": '0916_120920', "label": '88/40', "color": cmap(7), "linestyle":'-'}, # NT: 88/40
    {"configuration_name": 'diamond_normal_trainlist', "run_string": '0917_105029', "label": '80/32', "color": cmap(5), "linestyle":'-'}, # NT: 80/32
    {"configuration_name": 'diamond_normal_trainlist', "run_string": '0917_153719', "label": '72/24', "color": cmap(8), 'linestyle': '-'}, # NT: 72/24
    {"configuration_name": 'diamond_normal_trainlist', "run_string": '0918_150243', "label": '64/16', "color": cmap(6), 'linestyle': '-'}, # target model nT 64/16
    {"configuration_name": 'diamond_normal_trainlist', "run_string": '0921_194111', "label": '56/8', "color": cmap(3), "linestyle":'-'}, # NT: 56/8
    {"configuration_name": 'diamond_normal_trainlist', "run_string": '0923_131901', "label": '48/8', "color": cmap(4), "linestyle":'-'}, # NT: 48/8
]
dict_list4 = [
    {"configuration_name": 'diamond_TS_ALLD', "run_string": '0822_095336', "label": '256/64', "color": cmap(5), "linestyle":'-'}, # LinD: 256/64
    {"configuration_name": 'diamond_TS_ALLD', "run_string": '0819_100049', "label": '128/32', "color": cmap(5), "linestyle":'-'}, # LinD: 128/32
    {"configuration_name": 'diamond_TS_ALLD', "run_string": '0817_201010', "label": '88/40', "color": cmap(5), "linestyle":'-'}, # LinD: 88/40
    {"configuration_name": 'diamond_TS_ALLD', "run_string": '0817_200225', "label": '80/32', "color": cmap(5), "linestyle":'-'}, # LinD: 80/32
    {"configuration_name": 'diamond_TS_ALLD', "run_string": '0824_173255', "label": '72/24', "color": cmap(5), "linestyle":'-'}, # LinD: 72/24
    {"configuration_name": 'diamond_TwoSteps', "run_string": '0808_153800', "label": '64/16', 'color': cmap(5), 'linestyle': '-'}, # target model MAE between lin layer b:0805_103641
    {"configuration_name": 'diamond_TS_ALLD', "run_string": '0813_190135', "label": '56/8', "color": cmap(5), "linestyle":'-'}, # LinD: 56/8
    {"configuration_name": 'diamond_TS_ALLD', "run_string": '0816_151839', "label": '48/8',  "color": cmap(5), "linestyle":'-'}, # LinD: 48/8
]

test_config = read_json(parentfolder + '/test_configs/ALLD_test.json')

plot_string = 'MACSoverPESQ-0dB_' # UMBENNEN!

result_list = []
yerr = []
macs = [4639125504,
        2319562752,
        1416764928,
        1116358656,
        835073792,
        629234688,
        442516992,
        343039488]

model_LinD = pd.DataFrame()
for dnn_dict in dict_list1:
    configuration_name = dnn_dict["configuration_name"]
    print(configuration_name)
    run_string = dnn_dict["run_string"]


    run_result_folder = RESULTS_PATH.joinpath(configuration_name, run_string)
    results_fname = run_result_folder.joinpath('results.npy')

    # array has dimensions:
    #   (len(self.dataloader), len(self.metric_functions), len(self.SNR_list))
    results_array = np.load(results_fname, allow_pickle=True)
    print(results_array.shape)
    model_LinD[dnn_dict["label"]] = results_array[:,1,1]

    # Read metrics and SNRs corresponding to the results array
    #metric_functions = [getattr(metrics, met)() for met in test_config['metrics']]
    SNR_list = test_config['tester']['args']['SNR_list']

    #print(results_array.shape)
    result_list.append(np.median(results_array, axis=0)) # median over batches
    yerr.append(np.var(results_array, axis=0))
    #macs.append(int(dnn_dict["MACs"]))

all_results = np.stack(result_list, axis=2) # shape: (snrs, metrics, algorithms)
all_yerr = np.stack(yerr, axis=2)
pesq_result1 = all_results[:,1,:]
pesq_yerr1 = all_yerr[:,1,:]
yerr = []
result_list = []
    
model_Multi = pd.DataFrame()
for dnn_dict in dict_list2:
    configuration_name = dnn_dict["configuration_name"]
    print(configuration_name)
    run_string = dnn_dict["run_string"]

    run_result_folder = RESULTS_PATH.joinpath(configuration_name, run_string)
    results_fname = run_result_folder.joinpath('results.npy')

    # array has dimensions:
    #   (len(self.dataloader), len(self.metric_functions), len(self.SNR_list))
    results_array = np.load(results_fname, allow_pickle=True)
    print(results_array.shape)
    model_Multi[dnn_dict["label"]] = results_array[:,1,1]

    # Read metrics and SNRs corresponding to the results array
    #metric_functions = [getattr(metrics, met)() for met in test_config['metrics']]
    #SNR_list = test_config['tester']['args']['SNR_list']

    #print(results_array.shape)
    result_list.append(np.median(results_array, axis=0)) # median over batches
    yerr.append(np.var(results_array, axis=0))
    #macs.append(int(dnn_dict["MACs"]))

all_results = np.stack(result_list, axis=2) # shape: (snrs, metrics, algorithms)
all_yerr = np.stack(yerr, axis=2)
pesq_result2 = all_results[:,1,:]
pesq_result2[1,-1] = float('nan')
pesq_yerr2 = all_yerr[:,1,:]
pesq_yerr2[1,-1] = float('nan')
yerr = []
result_list = []

model_ohneKD = pd.DataFrame() 
for dnn_dict in dict_list3:
    configuration_name = dnn_dict["configuration_name"]
    print(configuration_name)
    run_string = dnn_dict["run_string"]

    run_result_folder = RESULTS_PATH.joinpath(configuration_name, run_string)
    results_fname = run_result_folder.joinpath('results.npy')

    # array has dimensions:
    #   (len(self.dataloader), len(self.metric_functions), len(self.SNR_list))
    results_array = np.load(results_fname, allow_pickle=True)
    print(results_array.shape)
    model_ohneKD[dnn_dict["label"]] = results_array[:,1,1]

    # Read metrics and SNRs corresponding to the results array
    #metric_functions = [getattr(metrics, met)() for met in test_config['metrics']]
    #SNR_list = test_config['tester']['args']['SNR_list']

    #print(results_array.shape)
    result_list.append(np.median(results_array, axis=0)) # median over batches
    yerr.append(np.var(results_array, axis=0))
    #macs.append(int(dnn_dict["MACs"]))

all_results = np.stack(result_list, axis=2) # shape: (snrs, metrics, algorithms)
all_yerr = np.stack(yerr, axis=2)
pesq_result3 = all_results[:,1,:]
pesq_yerr3 = all_yerr[:,1,:]

yerr = []
result_list = []
    
model_LSTM2 = pd.DataFrame()
for dnn_dict in dict_list4:
    configuration_name = dnn_dict["configuration_name"]
    print(configuration_name)
    run_string = dnn_dict["run_string"]

    run_result_folder = RESULTS_PATH.joinpath(configuration_name, run_string)
    results_fname = run_result_folder.joinpath('results.npy')

    # array has dimensions:
    #   (len(self.dataloader), len(self.metric_functions), len(self.SNR_list))
    results_array = np.load(results_fname, allow_pickle=True)
    print(results_array.shape)
    model_LSTM2[dnn_dict["label"]] = results_array[:,1,1]

    # Read metrics and SNRs corresponding to the results array
    #metric_functions = [getattr(metrics, met)() for met in test_config['metrics']]
    #SNR_list = test_config['tester']['args']['SNR_list']

    #print(results_array.shape)
    result_list.append(np.median(results_array, axis=0)) # median over batches
    yerr.append(np.var(results_array, axis=0))
    #macs.append(int(dnn_dict["MACs"]))

all_results = np.stack(result_list, axis=2) # shape: (snrs, metrics, algorithms)
all_yerr = np.stack(yerr, axis=2)
pesq_result4 = all_results[:,1,:]
pesq_yerr4 = all_yerr[:,1,:]

# save the dataframes
model_LinD.to_csv(out_folder.joinpath('model_LinD.csv'),index=False)
model_Multi.to_csv(out_folder.joinpath('model_Multi.csv'),index=False)
model_ohneKD.to_csv(out_folder.joinpath('model_ohneKD.csv'),index=False)
model_LSTM2.to_csv(out_folder.joinpath('model_LSTM2.csv'),index=False)

plot_fname = out_folder.joinpath(plot_string+'.png')

plt.figure(figsize=(8, 6))
plt.ylabel('PESQ')
plt.xlabel('MACs')
plt.grid()
for n_snr in range(len(SNR_list)):
    if SNR_list[n_snr] == 0:
        plt.errorbar(
            macs,
            pesq_result1[n_snr,:],
            pesq_yerr1[n_snr,:],
            label=f'KD Lin',#f'{SNR_list[n_snr]} dB', 
            linestyle='-', 
            marker='o', 
            color=cmap(5),
            capsize=5
            )
        plt.errorbar(
            macs,
            pesq_result2[n_snr,:],
            pesq_yerr2[n_snr,:],
            label=f'KD LSTM 1 + 2 + Lin',#f'{SNR_list[n_snr]} dB', 
            linestyle='-', 
            marker='o', 
            color=cmap(6),
            capsize=5
            )
        plt.errorbar(
            macs,
            pesq_result3[n_snr,:],
            pesq_yerr3[n_snr,:],
            label=f'without KD',#f'{SNR_list[n_snr]} dB', 
            linestyle='-', 
            marker='o', 
            color=cmap(2),
            capsize=5
            )
        plt.errorbar(
            macs,
            pesq_result4[n_snr,:],
            pesq_yerr4[n_snr,:],
            label=f'KD LSTM 2',#f'{SNR_list[n_snr]} dB',
            linestyle='-',
            marker='o',
            color=cmap(3),
            capsize=5
        )
        # plot a vertical blue line at the x position of the 1116358656 MACs
        plt.axvline(x=1116358656, color='b', linestyle='--')
        # plot a horizontal green line at the y position of pesq equal to 1.55
        plt.axhline(y=1.55, color='g', linestyle='--',label = 'teachers pesq-score at 0 dB SNR')

plt.legend(ncol=2)
plt.savefig(plot_fname)





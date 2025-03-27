# imports
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import torch.nn.utils.prune as prune
import torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import models
import json

from torchinfo import summary
from test import init_test_object, loadModel, loadParallelModel
from utils.json_handling import read_json
from thop import profile

parent_dir = '/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/2023_binauralspeechenhancement/'
RESULTS_PATH = pathlib.Path(os.path.join(parent_dir,'saved','test_output'))
MODELS_PATH = pathlib.Path(os.path.join(parent_dir,'saved','models'))
processing_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dictionary of the interesting models
dict_list = [
    {"configuration_name": "diamond_TS_ALLD", "run_string": "0817_200225","Info":"FT_JNF Original"}
]

# get the summary of one model
# construct the config_filename
for dnn in dict_list:
    configuration_name = dnn['configuration_name']
    print(configuration_name)
    run_string = dnn['run_string']
    model_info = dnn['Info']

    config_filename = MODELS_PATH.joinpath(configuration_name,run_string,'config.json')

    # read the config
    config = read_json(config_filename)

    # initialize the model
    model = init_test_object(test_config=config, 
                                object_name='model',
                                object_class=models).to(processing_device)

    # load the weights
    checkpoint_file = pathlib.Path(config_filename).parent.joinpath('model_best.pth')
    print(f'loading model: {checkpoint_file}')

    #try:
    #    model = loadModel(model, checkpoint_path=checkpoint_file, device=processing_device)
    #except:
    #    model = loadParallelModel(model=model, checkpoint_path=checkpoint_file, device=processing_device)

    print(f'[MODEL INFO] {model_info}')
    print(model)
    summary(model, input_size=(1, 10, 512))

    # random tensor
    input_tensor = torch.randn(32, 10, 512).to(processing_device)
    #x, mask = model(input_tensor, return_mask=True)

    # print mask values
    #print(mask.shape)+

    macs, params = profile(model, inputs=(input_tensor, ))
    print(f'[MACS INFO] {macs}')
    print(f'[PARAMS INFO] {params}')
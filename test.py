"""
Actual testing script.
This is intended to be called with a config file of a trained system, like for example

`python test.py -c saved/models/ff3_unet_iem/0310_132813/config.json -t test_configs/not_real_config.json`

"""
import argparse
import pathlib
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

import torch

import models
import datasets
import dataloaders
import testers
import metrics
from utils.json_handling import read_json


PROCESSING_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {PROCESSING_DEVICE}')


def init_test_object(test_config, object_name, object_class, **kwargs):

    # create a dictionary from the config file sub-directory
    object_args = dict(test_config[object_name]['args'])

    # add keyword args to arguments from dict
    object_args.update(kwargs)

    # instantiate the object with the object arguments
    object = getattr(object_class, test_config[object_name]['type'])(**object_args)

    return object


def loadModel(model, checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # load architecture params from checkpoint.
    model.load_state_dict(checkpoint['state_dict'])

    return model


def loadParallelModel(model, checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # load architecture params from checkpoint.
    model.module.load_state_dict(checkpoint['state_dict'])

    return model


def main(config: dict, config_filename: pathlib.Path, test_config_filename: pathlib.Path):

    test_config = read_json(test_config_filename)
    parent_dir = pathlib.Path(__file__).parent

    # from config file, infer the save path for results and create a subfolder

    model_name = config['experiment_name']

    if config['model']['type'] == 'NoisyInputDummy' or config['model']['type'] == 'NoisySTFTDummy':
        run_name = 'default_run'
    else:
        config_folder = config_filename.parent
        run_name = config_folder.name

    output_path = parent_dir / 'saved/test_output'
    test_output_folder = pathlib.Path(output_path, model_name, run_name)
    test_output_folder.parent.mkdir(exist_ok=True) # subfolder for model type / config type
    test_output_folder.mkdir(exist_ok=True)

    # create new model object
    model = init_test_object(test_config=config, object_name='model', object_class=models).to(PROCESSING_DEVICE)

    if not config['model']['type'] == 'NoisyInputDummy' and not config['model']['type'] == 'NoisySTFTDummy':

        model = torch.nn.DataParallel(model, device_ids=[0,1])

        # load weights
        checkpoint_file = os.path.join('saved', 'models', model_name, run_name, 'model_best.pth')
        print(f'loading model: {checkpoint_file}')

        try:
            model = loadModel(model, checkpoint_path=checkpoint_file, device=PROCESSING_DEVICE)
        except:
            model = loadParallelModel(model, checkpoint_path=checkpoint_file, device=PROCESSING_DEVICE)

        model.eval()

    test_dataset = init_test_object(test_config=test_config, object_name='test_dataset', object_class=datasets)
    test_dataloader = init_test_object(test_config=test_config, object_name='dataloader', object_class=dataloaders, dataset=test_dataset)

    metric_functions = [getattr(metrics, met)(device=PROCESSING_DEVICE) for met in test_config['metrics']]

    tester = init_test_object(test_config=test_config, object_name='tester', object_class=testers,
                              model=model,
                              metric_functions=metric_functions,
                              dataloader=test_dataloader,
                              config=config,
                              test_output_folder=test_output_folder,
                              device=PROCESSING_DEVICE
    )

    tester.test()


if __name__ == '__main__':
    parent_dir = pathlib.Path(__file__).parent
    config_path = parent_dir / 'saved/models/diamond_TwoSteps/1023_132130/config.json'
    test_config_path = parent_dir / 'test_configs/diamond_snr_test_array.json'
    print(f'config: {config_path}')
    print(f'test config: {test_config_path}')
    args = argparse.ArgumentParser(description='PyTorch training script')
    args.add_argument('-c', '--config', default=config_path, type=str,
                      help='config file path (default: None)')
    args.add_argument('-t', '--testconfig', default=test_config_path, type=str,
                      help='test config file path (default: None)')

    opts = args.parse_args()
    config_filename = pathlib.Path(opts.config)
    test_config_filename = pathlib.Path(opts.testconfig)

    config = read_json(config_filename)

    with torch.no_grad():
        main(config, config_filename=config_filename, test_config_filename=test_config_filename)

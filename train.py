import argparse
import os
import datetime

#os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

import torch
import numpy as np

import models
import datasets
import dataloaders
import trainers

import losses
import metrics
from utils.config_parser import ConfigParser

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# use this with True if input sizes to the DNN do not vary (cudnn will select an optimal algorithm leading to faster runtime)
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
torch.backends.cudnn.benchmark = True

np.random.seed(SEED)


GPU_PROCESSING_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {GPU_PROCESSING_DEVICE}')

def main(config: ConfigParser):

    # create new model
    model = config.init_obj('model', models).to(GPU_PROCESSING_DEVICE)
    model = torch.nn.DataParallel(model, device_ids=[0,1]) # TODO: DistributedDataParallel?

    # if pretrained weights are available/set in config, load them
    if "path_pretrained_dict" in config["model"].keys():
        model.set_to_finetune_mode(config["model"]["path_pretrained_dict"])
        if not model.is_in_finetune_mode:
            # sanity check whether finetune mode was actually activated and the parameters loaded
            raise NotImplementedError('The model could not be properly loaded!')

    # initialize dataset, then dataloader for training
    train_dataset = config.init_obj('dataset', datasets)
    train_dataloader = config.init_obj('dataloader', dataloaders, dataset=train_dataset)

    # initialize dataset, then dataloader for validation
    validation_dataset = config.init_obj('validation_dataset', datasets)
    validation_dataloader = config.init_obj('dataloader', dataloaders, dataset=validation_dataset)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    criterion = getattr(losses, config['loss'])(device=GPU_PROCESSING_DEVICE)
    metric_functions = [getattr(metrics, met)() for met in config['metrics']]

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = config.init_obj('trainer', trainers,
        model=model,
        criterion=criterion,
        metric_functions=metric_functions,
        optimizer=optimizer,
        dataloader=train_dataloader,
        valid_dataloader=validation_dataloader,
        lr_scheduler=lr_scheduler,
        config=config,
        device=GPU_PROCESSING_DEVICE)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch training script')
    args.add_argument('-c', '--config', default="configs/diamond_twoSteps_config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)') # TODO: not being used!
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)') # TODO: not being used!

    start_time = datetime.datetime.now()
    print(f'start_time: {start_time}')

    config = ConfigParser.from_args(args)
    main(config)

    stop_time = datetime.datetime.now()
    print(f'stop_time: {stop_time}')

    print(f'total duration: {stop_time-start_time}')

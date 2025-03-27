import torch
import torch.nn.utils.prune as prune
import copy
import os
import sys
import pathlib
import argparse
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import models

from utils.config_parser import ConfigParser
from torchinfo import summary
from utils.inference_speed import inference_speed


def MyPrune(model: torch.nn.Module, kwargs: dict = None):
    """Performs a global unstructered pruning of a FT_JNF model."""

    print("[INFO] Pruning model")
    
    # create a deep copy of the model
    pruned_model = copy.deepcopy(model)

    # check if config got a key for the pruning
    if kwargs is None:
        print("[WARNING] No pruning parameters given. Using default values.")
        kwargs = {'method': 'structured',
                  'amount': 0.2, 
                  'save_flag': False,
                  'parameters_to_prune': (
                        (pruned_model.lstm1, "weight_ih_l0"),
                        (pruned_model.lstm1, "weight_hh_l0"),
                        #(pruned_model.lstm1, "bias_ih_l0"),
                        #(pruned_model.lstm1, "bias_hh_l0"),
                        (pruned_model.lstm2, "weight_ih_l0"),
                        (pruned_model.lstm2, "weight_hh_l0"),
                        #(pruned_model.lstm2, "bias_ih_l0"),
                        #(pruned_model.lstm2, "bias_hh_l0"),
                        (pruned_model.ff, "weight"),
                        #(pruned_model.ff, "bias"),
                    ),
                    'review': True,
                    'calc_prune_amount': False,
                  }
    print(f"[INFO] Using pruning parameters: {kwargs}")

    # calculate the amount of parameters to prune
    if kwargs['calc_prune_amount']:
        # calc how many parameters in the model are equal to zero
        assert 'Not Implemented Yet!'

    # prune the model
    if kwargs['method'] == 'global_unstructured':
        prune.global_unstructured(
            kwargs['parameters_to_prune'],
            pruning_method=prune.L1Unstructured,
            amount=kwargs['amount'],
            )
    elif kwargs['method'] == 'structured':
        for module, name in kwargs['parameters_to_prune']:
            prune.ln_structured(module, name, amount=kwargs['amount'], n=2, dim=0)
    else:
        raise ValueError(f"Pruning method {kwargs['method']} not supported.")
    
    # remove the pruning re-parametrization
    for module, name in kwargs['parameters_to_prune']:
        prune.remove(module, name)

    if kwargs['save_flag']:
        parent_dir = pathlib.Path(__file__).parent.parent
        # build directory for temporal storage
        temp_storage_dir = parent_dir / "saved/temp_stored_model"
        # check if directory exists, if not create it
        if not os.path.exists(temp_storage_dir):
            os.makedirs(temp_storage_dir)
        # save the pruned model to the temporal storage
        print(f"[INFO] Saving pruned model to {temp_storage_dir}/temp_pruned_model.pth")
        torch.save(pruned_model.state_dict(), temp_storage_dir / "temp_pruned_model.pth")

    # show review of the model and the pruned model
    if kwargs['review']:
        zero_counter = 0 
        nelements = 0
        for module, name in kwargs['parameters_to_prune']:
            print(f"Sparsity in {name}: {100. * float(torch.sum(module.__getattr__(name) == 0)) / float(module.__getattr__(name).nelement())}%")
            zero_counter += torch.sum(module.__getattr__(name) == 0)
            nelements += module.__getattr__(name).nelement()
        print(f"Global sparsity: {100. * float(zero_counter) / float(nelements)}%")
    
        # test inference speed
        print("[INFO] Testing inference speed")
        test_tensor = torch.randn(1,5,32000)
        print(f"Inference speed before pruning: {inference_speed(model, test_tensor)}")
        print(f"Inference speed after pruning: {inference_speed(pruned_model, test_tensor)}")

        # show summary of the model
        print("[INFO] Model summary before pruning")
        summary(model, input_size=(1,5,32000))
        print("[INFO] Model summary after pruning")
        summary(pruned_model, input_size=(1,5,32000))

    return pruned_model

if __name__ == '__main__':
    '''Test script for the pruning function.'''
    parent_dir = pathlib.Path(__file__).parent.parent
    args = argparse.ArgumentParser(description='PyTorch pruning function')
    args.add_argument('-c', '--config', default=parent_dir/"configs/diamond_5mic_reverb_config.json")
    args.add_argument('-r', '--resume', default=None, 
                      type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, 
                      type=str, help='indices of GPUs to enable (default: all)')
    
    # handle config file and init model for test
    config = ConfigParser.from_args(args)
    model = config.init_obj('model', models)
 
    pruned_model = MyPrune(model)


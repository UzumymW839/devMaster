
import pathlib
import torch
import torchaudio
import sys
import os
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import models
from test import init_test_object, loadModel, loadParallelModel
from utils.json_handling import read_json
from transforms.normalization_transforms import Normalization



def main():

    config_filename = pathlib.Path(__file__).parent.parent / 'saved/models/diamond_5mic_real/1010_135125/config.json'

    processing_device = torch.device('cpu')

    config = read_json(config_filename)

    model = init_test_object(test_config=config, object_name='model', object_class=models).to(processing_device)


    # load weights
    checkpoint_file = pathlib.Path(config_filename).parent.joinpath('model_best.pth')
    print(f'loading model: {checkpoint_file}')

    try:
        model = loadModel(model, checkpoint_path=checkpoint_file, device=processing_device)
    except:
        model = loadParallelModel(model, checkpoint_path=checkpoint_file, device=processing_device)

    model.eval()

    # get the path of the current folder
    parent_folder = pathlib.Path(__file__).parent.parent
    # example_folder = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE2/ohlems/work/diamond_data/real_diamond_examples')
    example_folder = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/real_diamond_examples/records')
    # example_folder = pathlib.Path(parent_folder.joinpath('saved/models/reverb_experiment/0313_160814/best_audio_examples'))
    select_oem_ch_idxs = [0, 1, 2, 3, 4]

    for noisy_fname in example_folder.glob('*.wav'):#[example_folder.joinpath('5mic_array_testaufnahmen.wav')]:#example_folder.glob('*.wav'): #[example_folder.joinpath('noisy input.wav')]: # #example_folder.iterdir():

        noisy_data, fs_noisy = torchaudio.load(noisy_fname)
        if fs_noisy != 16000:
            noisy_data = torchaudio.functional.resample(noisy_data, orig_freq=fs_noisy, new_freq=16000)
        noisy_data = noisy_data[select_oem_ch_idxs, ...]

        norm = Normalization(reference_channel_target=0)

        x, _ = norm(noisy_data=noisy_data, target_data=noisy_data)

        y, mask = model(x=x[None, ...], return_mask=True)
        #y = model(x=x[None, ...])[0,...] # unsqueeze batch dim
        print(x[None,...].shape)
        #print(x[None, :, 16000*8:].shape)
        #y[:, 16000*8:] = model(x=x[None, :, 16000*8:], return_mask=False)[0,...]


        x /= torch.amax(torch.abs(x))
        y /= torch.amax(torch.abs(y))
        y = y.squeeze(dim=0)
        # print the curretn folder
        # save_path = pathlib.Path('/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/2023_binauralspeechenhancement/saved/diamond_examples')
        save_path = pathlib.Path(parent_folder.joinpath('saved/diamond_examples'))
        torchaudio.save(save_path.joinpath(f'proc_{noisy_fname.name}'), y, 16000)
        torchaudio.save(save_path.joinpath(f'unproc_{noisy_fname.name}'), x, 16000)
        print(f'processed {noisy_fname}')


if __name__ == '__main__':
    main()

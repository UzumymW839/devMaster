
import pathlib
import torch
import torchaudio

import models
from test import init_test_object, loadModel, loadParallelModel
from utils.json_handling import read_json
from transforms.normalization_transforms import Normalization



def main():

    #config_filename = r'saved/models/default_experiment/0717_132739/config.json'
    #config_filename = r'saved/models/default_experiment/0804_162746/config.json'
    #config_filename = r'saved/models/reverb_experiment/0731_135214/config.json'
    config_filename = r'saved/models/default_experiment/0814_103021/config.json' # 1sec trail cutoff
    #config_filename = r'saved/models/binaural_experiment/0815_154712/config.json' # binaural without trail cutoff and without reverb
    config_filename = r'saved/models/binaural_experiment/0821_114728/config.json' # binaural with trail cutoff and with reverb

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

    #example_folder = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/ohlems/work/BSE_data/real_recordings')
    example_folder = pathlib.Path(r'/daten_pool/ohlems/BSE_data/real_recordings_2')
    select_oem_ch_idxs = [1,3]

    for noisy_fname in example_folder.iterdir(): #[example_folder.joinpath('noisy_0grad.wav')]: #example_folder.iterdir():

        noisy_data, fs_noisy = torchaudio.load(noisy_fname)
        noisy_data = torchaudio.functional.resample(noisy_data, orig_freq=fs_noisy, new_freq=16000)
        noisy_data = noisy_data[select_oem_ch_idxs, ...]

        norm = Normalization(reference_channel_target=1)

        x, _ = norm(noisy_data=noisy_data, target_data=noisy_data)

        #y = model(x=x[None, ...], return_mask=False)[0,...]
        y = model(x=x[None, ...])[0,...]
        print(x[None,...].shape)
        #print(x[None, :, 16000*8:].shape)
        #y[:, 16000*8:] = model(x=x[None, :, 16000*8:], return_mask=False)[0,...]


        x /= torch.amax(torch.abs(x))
        y /= torch.amax(torch.abs(y))
        torchaudio.save(f'saved/example_audio_processed/{noisy_fname.name}', y, 16000)
        torchaudio.save(f'saved/example_audio_unprocessed/{noisy_fname.name}', x, 16000)
        print(f'processed {noisy_fname}')


if __name__ == '__main__':
    main()

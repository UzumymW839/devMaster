import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch
import models
import datasets
import dataloaders

from test import init_test_object, loadModel, loadParallelModel
from utils.json_handling import read_json
from transforms.normalization_transforms import Normalization
from testers.real_noise_tester import RealNoiseTester
from utils.config_parser import ConfigParserTest
from utils.training_utils import save_maxscaled_audio

def main():
    # define the model config file
    parent_folder = pathlib.Path(__file__).parent.parent
    config_filenames = [
        parent_folder / 'saved/models/diamond_5mic_experiment/0801_143408/config.json', # teacher model
        # parent_folder / 'saved/models/diamond_TwoSteps/0626_130824/config.json', # target model with KD MAE with Mask
        parent_folder / 'saved/models/diamond_LinD/0828_232658/config.json', # target model with KD MAE between all layers
    ]

    processing_device = torch.device('cpu')

    # create dataset (use reverberated frontal speech dataset with file list diamond reverberated frontal speech dataset)
    file_list = pathlib.Path(__file__).parent.parent / 'file_lists/diamond_reverberated_multichannel_speech/test.csv'
    test_dataset = datasets.ReverberatedFrontalSpeechDataset(
            file_list_filename=file_list,
            noise_transform="NoiseTransform",
            normalization_transform="Normalization",
            noise_file_list_filename=pathlib.Path(__file__).parent.parent / 'file_lists//diamond_multichannel_noise/training.csv',
            noise_fold="/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/multichannel_noise",
        )
    print(f'[INFO] loaded dataset with {len(test_dataset)} examples')
    print(test_dataset)

    # create dataloader
    test_dataloader = dataloaders.BaseDataLoader(test_dataset,
                                                    batch_size=1,
                                                    num_workers=1,
                                                    )
        
    print(f'[INFO] loaded dataloader with {len(test_dataloader)} batches')
    print(test_dataloader)

    # run test send one item from dataloader to the model
    data, target = next(iter(test_dataloader))
    
    for idx in range(len(config_filenames)):
        config = read_json(config_filenames[idx])
        model = init_test_object(test_config=config, object_name='model', object_class=models).to(processing_device)

        # load weights
        checkpoint_file = pathlib.Path(config_filenames[idx]).parent.joinpath('model_best.pth')
        print(f'loading model: {checkpoint_file}')

        try:
            model = loadModel(model, checkpoint_path=checkpoint_file, device=processing_device)
        except:
            model = loadParallelModel(model, checkpoint_path=checkpoint_file, device=processing_device)

        model.eval()

        output = model(data)

        with torch.no_grad():
            inp = data.detach().cpu().squeeze()
            tar = torch.squeeze(target.detach().cpu(), dim=0)
            out = torch.squeeze(output.detach().cpu(), dim=0)

            # save results in saved/diamond_examples as wav files
            parent_path = pathlib.Path(__file__).parent.parent
            output_path = parent_path / 'saved/diamond_examples'
            print(f'[INFO] saved example files in {output_path}')

            audio_name_in = "random_example_input" + str(idx)
            audio_name_tar = "random_example_target" + str(idx)
            audio_name_out = "random_example_output" + str(idx)

            # save single and multichannel audio files
            save_maxscaled_audio(inp, audio_name_tag=audio_name_in, audio_example_path=output_path)
            save_maxscaled_audio(tar, audio_name_tag=audio_name_tar, audio_example_path=output_path)
            save_maxscaled_audio(out, audio_name_tag=audio_name_out, audio_example_path=output_path)

if __name__ == '__main__':
    main()

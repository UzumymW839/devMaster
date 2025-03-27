import pathlib
import datasets
import dataloaders
from dataloaders.collate_functions import mix_collate_fn
from utils.training_utils import save_maxscaled_audio



def main():

    # initialize dataset, then dataloader for training
    train_dataset = datasets.BinauralDataset(
        file_list_filename = "file_lists/clean_multichannel_speech/training.csv",
        noise_transform = "NoiseTransform",
        normalization_transform = "Normalization",
        noise_file_list_filename = "file_lists/multichannel_noise/training.csv",
        noise_fold = "/daten_pool/ohlems/BSE_data/multichannel_noise"
    )

    #train_dataloader = dataloaders.DataLoader(
    #    dataset=train_dataset,
    #    collate_fn=mix_collate_fn,
    #    num_workers=8,
    #    batch_size=4,
    #    drop_last=True
    #)

    #batch_noisy, batch_target = next(iter(train_dataloader))

    #noisy = batch_noisy[0, ...]
    #target = batch_target[0, ...]
    noisy, target = next(iter(train_dataset))

    out_fold = pathlib.Path('saved/debug_binaural_audio')
    save_maxscaled_audio(wav_data=noisy, audio_example_path=out_fold, audio_name_tag='noisy', samplerate=16000)
    save_maxscaled_audio(wav_data=target, audio_example_path=out_fold, audio_name_tag='target', samplerate=16000)

if __name__ == '__main__':
    main()

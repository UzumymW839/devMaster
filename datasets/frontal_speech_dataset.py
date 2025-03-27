import pathlib
from typing import Union
import torch
import torchaudio

from datasets.base_dataset import BaseDataset
from transforms import noise_transforms, normalization_transforms
from utils.audio_utils import random_fill_or_cut_speech_audio, torch_rms

DEFAULT_CONFIG = {
    'use_channels': [0, 1], # Left outside, right outside
    'reference_channel_target': 1, # indexing based on use_channels entries -> left outside
    'example_length_seconds': 4,
    'desired_samplerate': 16000
}

BINAURAL_CONFIG = DEFAULT_CONFIG.copy()
BINAURAL_CONFIG['reference_channel_target'] = [0, 1]
BINAURAL_CONFIG['reference_channel_snr'] = 1

class FrontalSpeechDataset(BaseDataset):

    def __init__(self, file_list_filename,
                 configuration=DEFAULT_CONFIG,
                 #reference_channel_target=[1], # indexing based on use_channels entries -> left outside
                 #reference_channel_normalization=None,
                 noise_transform=None,
                 normalization_transform=None,
                 noise_file_list_filename=None,
                 noise_ignore_channel=None,
                 noise_fold=None,
                 cut_end=False,
                 snr_db: Union[float, list]=[0, 30],
                 use_channels=None,
                 ) -> None:
        super().__init__(file_list_filename, configuration)

        if use_channels is None:
            self.use_channels = configuration['use_channels']
        else:
            self.use_channels = use_channels
        self.example_length_seconds = configuration['example_length_seconds']
        self.desired_samplerate = configuration['desired_samplerate']
        self.reference_channel_target = configuration['reference_channel_target']
        #self.reference_channel_target = reference_channel_target

        #if reference_channel_normalization is None:
        #    self.reference_channel_normalization = self.reference_channel_target[0]
        #else:
        #    self.reference_channel_normalization = reference_channel_normalization

        if noise_transform is not None:
            self.noise_transform = getattr(noise_transforms, noise_transform)(
                desired_samplerate=self.desired_samplerate,
                reference_channel_target=self.reference_channel_target,
                example_length_seconds=self.example_length_seconds,
                file_list_filename=noise_file_list_filename,
                ignore_channel=noise_ignore_channel,
                noise_fold=noise_fold,
                snr_db=snr_db
            )
        else:
            self.noise_transform = None

        if normalization_transform is not None:
            self.normalization_transform = getattr(normalization_transforms, normalization_transform)(
                #reference_channel_target=self.reference_channel_normalization
                reference_channel_target=self.reference_channel_target
            )
        else:
            self.normalization_transform = None

        self.cut_end = cut_end



    def get_full_example(self, index: int):
        """
        CVDE can also have 'empty' speech files.
        If one is detected, another one is drawn at random.
        """
        speech_filename = pathlib.Path(self.filenames_dataframe['file_name'][index])
        audio_data, samplerate = torchaudio.load(speech_filename, channels_first=True)

        if (torch_rms(audio_data, axis=-1) == 0).any():
            #print(f"weird speech file in CVDE dataset! filename was: {speech_filename}, now drawing another random file")
            # get another random example
            random_index = int(torch.randint(low=0, high=len(self.filenames_dataframe['file_name']), size=[1]))
            audio_data, samplerate, speech_filename = self.get_full_example(random_index)

        return audio_data, samplerate, speech_filename



    def __getitem__(self, index: int):

        # get full input audio
        audio_data, samplerate, speech_filename = self.get_full_example(index=index) # (channels, time)

        audio_data = audio_data[self.use_channels, :]

        # random time selection or padding
        length_samples = self.example_length_seconds * samplerate
        target_data = random_fill_or_cut_speech_audio(audio_data, length_samples=length_samples)

        if self.noise_transform is not None:
            noisy_data = self.noise_transform(target_data, speech_filename=speech_filename)
        else:
            noisy_data = target_data.clone()

        target_data = target_data[[self.reference_channel_target], :]

        if self.normalization_transform is not None:
            noisy_data, target_data = self.normalization_transform(noisy_data, target_data)

        if self.cut_end:
            noisy_data = noisy_data[:, :-self.desired_samplerate]
            target_data = target_data[:, :-self.desired_samplerate]

        assert not torch.isnan(target_data).any(), f'NaN data found for file: {speech_filename}'

        # return input data and label
        return noisy_data, target_data



class BinauralDataset(FrontalSpeechDataset):
    def __init__(self, configuration=BINAURAL_CONFIG, **kwargs):
        super().__init__(configuration=configuration, **kwargs)

        self.reference_channel_snr = configuration['reference_channel_snr']

        if self.noise_transform is not None:
            # override reference channel setting because in the binaural dataset, these can be multiple channels
            # but the SNR reference should only be one
            self.noise_transform.reference_channel_target = self.reference_channel_snr


    def __getitem__(self, index: int):

        # get full input audio
        audio_data, samplerate, speech_filename = self.get_full_example(index=index) # (channels, time)

        # random time selection or padding
        length_samples = self.example_length_seconds * samplerate
        target_data = random_fill_or_cut_speech_audio(audio_data, length_samples=length_samples)

        if self.noise_transform is not None:
            noisy_data = self.noise_transform(target_data, speech_filename=speech_filename)
        else:
            noisy_data = target_data.clone()

        target_data = target_data[self.reference_channel_target, :]

        if self.normalization_transform is not None:
            noisy_data, target_data = self.normalization_transform(noisy_data, target_data)

        if self.cut_end:
            noisy_data = noisy_data[:, :-self.desired_samplerate]
            target_data = target_data[:, :-self.desired_samplerate]

        assert not torch.isnan(target_data).any(), f'NaN data found for file: {speech_filename}'

        # return input data and label
        return noisy_data, target_data


if __name__ == '__main__':

    file_list_filename = "file_lists/clean_multichannel_speech/training.csv"
    noise_transform = None #"NoiseTransform"
    normalization_transform = "Normalization"
    noise_file_list_filename = "file_lists/multichannel_noise/training.csv"
    noise_fold = "/daten_pool/ohlems/BSE_data/multichannel_noise"

    ds = BinauralDataset(
        file_list_filename=file_list_filename,
        noise_transform=noise_transform,
        normalization_transform=normalization_transform,
        noise_file_list_filename=noise_file_list_filename,
        noise_fold=noise_fold,
        #reference_channel_target=reference_channel_target
        )

    n, t = ds.__getitem__(0)
    #print(n.shape)
    print(t.shape)

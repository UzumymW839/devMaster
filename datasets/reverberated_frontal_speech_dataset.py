import pathlib
import torch
import torchaudio

from datasets.frontal_speech_dataset import FrontalSpeechDataset
from utils.audio_utils import random_fill_or_cut_speech_audio

# reverberated dataset is indexed as follows:
# [0,1] left outside, right outside (both reverberated)
# [2,3] left outside, right outside (both anechoic)
REVERB_CONFIG = {
    'use_channels': [0, 1, 2, 3, 4], # Left outside, right outside
    'reference_channel_target': 0, # indexing based on use_channels entries -> left outside reverberated
    'select_channel_target': 5, # indexing based on use_channels entries -> left outside anechoic
    'example_length_seconds': 4,
    'desired_samplerate': 16000
}


BINAURAL_REVERB_CONFIG = REVERB_CONFIG.copy()
BINAURAL_REVERB_CONFIG['reference_channel_target'] = [0, 1]
BINAURAL_REVERB_CONFIG['reference_channel_snr'] = 1


class ReverberatedFrontalSpeechDataset(FrontalSpeechDataset):

    def __init__(self, configuration=REVERB_CONFIG, **kwargs) -> None:
        super().__init__(configuration=configuration, **kwargs)

        # different from FrontalSpeechDataset, this dataset uses select_channel_target to pick the anechoic target channel from
        # the full audio file and reference_channel_target for normalization within the noisy channel order (depends on use_channels)
        self.select_channel_target = configuration['select_channel_target']


    def __getitem__(self, index: int):

        # get full input audio
        audio_data, samplerate, speech_filename = self.get_full_example(index=index) # (channels, time)

        # random time selection or padding
        length_samples = self.example_length_seconds * samplerate
        target_data = random_fill_or_cut_speech_audio(audio_data, length_samples=length_samples)

        # use the reverberated speech channels as a basis for noise mixing
        if self.noise_transform is not None:
            noisy_data = self.noise_transform(target_data[self.use_channels, :], speech_filename=speech_filename)
        else:
            noisy_data = target_data[self.use_channels, :].clone()

        target_data = target_data[[self.select_channel_target], :]

        if self.normalization_transform is not None:
            noisy_data, target_data = self.normalization_transform(noisy_data, target_data)

        if self.cut_end:
            noisy_data = noisy_data[:, :-self.desired_samplerate]
            target_data = target_data[:, :-self.desired_samplerate]


        assert not torch.isnan(target_data).any(), f'NaN data found for file: {speech_filename}'

        # return input data and label
        return noisy_data, target_data



class BinauralReverbDataset(ReverberatedFrontalSpeechDataset):
    def __init__(self, configuration=BINAURAL_REVERB_CONFIG, **kwargs):
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


        # use the reverberated speech channels as a basis for noise mixing
        if self.noise_transform is not None:
            noisy_data = self.noise_transform(target_data[self.use_channels, :], speech_filename=speech_filename)
        else:
            noisy_data = target_data[self.use_channels, :].clone()

        target_data = target_data[self.reference_channel_target, :]

        if self.normalization_transform is not None:
            noisy_data, target_data = self.normalization_transform(noisy_data, target_data)

        if self.cut_end:
            noisy_data = noisy_data[:, :-self.desired_samplerate]
            target_data = target_data[:, :-self.desired_samplerate]

        assert not torch.isnan(target_data).any(), f'NaN data found for file: {speech_filename}'

        # return input data and label
        return noisy_data, target_data

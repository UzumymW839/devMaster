import collections
import pathlib
import pandas as pd
import torch
import torchaudio
from typing import Union

from utils.audio_utils import random_fill_or_cut_speech_audio, torch_rms


def snr(signal, noise):
    sigpow = torch_rms(signal, axis=1)
    noisepow = torch_rms(noise, axis=1)
    return 20 * torch.log10(sigpow / noisepow)

def mag2db(mag_val):
    return 20*torch.log10(mag_val)

def db2mag(db_val):
    return 10**(db_val/20)


class NoiseTransform(torch.nn.Module):

    def __init__(self,
                 file_list_filename: Union[pathlib.Path, str, None]=None,
                 reference_channel_target: int=0,
                 ignore_channel: Union[int, None]=None,
                 snr_db: Union[float, list]=[0, 30],
                 example_length_seconds: float=4,
                 desired_samplerate: int=16000,
                 noise_fold=None
        ) -> None:

        super().__init__()
        self.reference_channel_target = reference_channel_target
        self.example_length_seconds = example_length_seconds
        self.desired_samplerate = desired_samplerate

        if noise_fold is not None:
            self.noise_fold = pathlib.Path(noise_fold) # TODO: is this actually necessary?

        # if snr_db is a range, make sure its two values
        if isinstance(snr_db, collections.abc.Sequence):
            assert len(snr_db) == 2
        self.snr_db = snr_db

        if file_list_filename is None:
            raise ValueError("actually this does need a file list! huh")
        elif isinstance(file_list_filename, str):
            self.file_list_filename = pathlib.Path(file_list_filename)
        else:
            self.file_list_filename = file_list_filename
        print(self.file_list_filename)

        self.init_noise_list(file_list_filename=self.file_list_filename)

        self.ignore_channel = ignore_channel
        if self.ignore_channel is not None:
            print(f'zero-ing out channel in noise files: channel {self.ignore_channel}')


    def init_noise_list(self, file_list_filename: pathlib.Path):
        """
        Read list of noise files from CSV file into dataframe.
        """
        self.filenames_dataframe = pd.read_csv(file_list_filename)
        # early fail if problems with noise files
        assert len(self.filenames_dataframe) > 0
        #print(f'using {n_noise} noise recordings')


    def set_fixed_snr(self, snr_db: float):
        """
        Sets the SNR to a fixed, set value in decibels.
        This function can be called from outside to set up testing/training
        at a defined SNR level.
        """
        self.snr_db = snr_db


    def get_noise(self, speech_filename: pathlib.Path):
        """get random noise."""

        # get random filename
        random_index = torch.randint(low=0, high=len(self.filenames_dataframe), size=[1])
        noise_filename = self.filenames_dataframe['file_name'][int(random_index)]

        # get full input audio
        noise_data, samplerate = torchaudio.load(noise_filename, channels_first=True)

        # random time selection or padding
        length_samples = self.example_length_seconds * samplerate
        noise_data = random_fill_or_cut_speech_audio(noise_data, length_samples=length_samples)

        # resampling
        noise_data = torchaudio.functional.resample(waveform=noise_data,
                                              orig_freq=samplerate,
                                              new_freq=self.desired_samplerate)

        # if there are any channels to be ignored, do not add noise to them
        if self.ignore_channel is not None:
            noise_data[self.ignore_channel ,:] = 0

        return noise_data, noise_filename


    def get_random_snr_value(self):
        """
        Generate a random SNR value inbetween the limits given in
        self.snr_db, which is expected to be a two-element list here.
        The SNR values resulting from calling this function should have
        a uniform distribution on the dB scale.
        """

        # range limits
        low_db = self.snr_db[0]
        high_db = self.snr_db[1]

        # scale [0,1) to [0, high_db-low_db)
        scaling_factor = high_db - low_db

        # shift [0, high_db-low_db) to [low_db, high_db)
        offset = low_db

        # generate random value and transform it to given boundaries
        snr_db = scaling_factor * torch.rand(size=[1]) + offset

        return snr_db


    def get_desired_snr_db(self):
        """
        This function returns an SNR target for re-scaling
        noise during mixing speech and noise. The behavior
        is different depending on whether self.snr_db is a list
        or scalar.
        """

        if isinstance(self.snr_db, collections.abc.Sequence):
            # if SNR is a range rather than a fixed value,
            # draw an random SNR in dB
            desired_snr_db = self.get_random_snr_value()
        else:
            # it's a single fixed value, just propagate it
            desired_snr_db = self.snr_db

        return desired_snr_db


    def forward(self, x: torch.Tensor, speech_filename: pathlib.Path):

        # get a noise signal (multi-channel)
        # shape: (channels, time)
        # the get_noise() method is implemented differently per transform!
        noise, noise_filename = self.get_noise(speech_filename=speech_filename)

        # add a tiny bit of white noise
        noise += 1e-9 * torch.randn_like(noise) # shape: (channels, time)

        # current SNR at reference channel
        curr_snr_db = snr(signal=x[[self.reference_channel_target],:],
                        noise=noise[[self.reference_channel_target],:])

        # get a desired SNR value (either fixed or randomly chosen)
        desired_snr_db = self.get_desired_snr_db()

        # compute scaling factor necessary to arrive at desired SNR at reference microphone
        noise_scale_fac = db2mag(curr_snr_db - desired_snr_db)

        # add scaled noise and return noisy signal
        return x + noise * noise_scale_fac # shape: (channels, time)

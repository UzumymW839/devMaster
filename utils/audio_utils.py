import random
import numpy as np
import torch

from utils.vad import active_speech_level

def random_fill_or_cut_speech_audio(audio_data: torch.Tensor, length_samples: int) -> torch.Tensor:
    """
    Input audio_data assumed to have shape (channels, time) which is torchaudio default.
    If the input size in time dimension is longer than length_samples, a random selection is performed.
    If the input size in time dimension is shorter than length_samples, the signal is padded to the length
    by adding zeros in front or at the end (randomly chosen).
    """
    if audio_data.shape[1] > length_samples:
        # subselect signal
        subselect_start = random.randint(0, audio_data.shape[1]-length_samples)
        subselect_indices = torch.arange(start=subselect_start, step=1, end=subselect_start+length_samples)
        audio_data = audio_data[:, subselect_indices]

    elif audio_data.shape[1] < length_samples:
        # random padding
        audio_data_temp = audio_data.clone()
        audio_data = torch.zeros(audio_data.shape[0], length_samples)
        subselect_start = random.randint(0, length_samples - audio_data.shape[1])
        subselect_indices = torch.arange(start=subselect_start, step=1, end=subselect_start+audio_data_temp.shape[1])
        audio_data[:, subselect_indices] = audio_data_temp

    return audio_data


def random_cut_speech_audio(audio_data: torch.Tensor, length_samples: int) -> torch.Tensor:
    """
    Input audio_data assumed to have shape (channels, time) which is torchaudio default.
    If the input size in time dimension is longer than length_samples, a random selection is performed.
    """
    if audio_data.shape[1] > length_samples:
        # subselect signal
        subselect_start = random.randint(0, audio_data.shape[1]-length_samples)
        subselect_indices = torch.arange(start=subselect_start, step=1, end=subselect_start+length_samples)
        audio_data = audio_data[:, subselect_indices]

    return audio_data


def torch_rms(x: torch.Tensor, axis: int=0):
    """
    Root mean square of x alongside axis axis.
    """
    return torch.sqrt(torch.mean(torch.square(x), axis=axis))

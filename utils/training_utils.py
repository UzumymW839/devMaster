import pathlib
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt


EXAMPLE_SAMPLING_FREQUENCY = 16000
EXAMPLE_SPECTROGRAM_NFFT = 512


def save_maxscaled_audio(wav_data: torch.Tensor, audio_example_path: pathlib.Path,
                         audio_name_tag: str, samplerate: int=EXAMPLE_SAMPLING_FREQUENCY):

    # make sure wav is in range[-1, 1] before saving
    wav_data = wav_data / wav_data.abs().max()

    # write audio
    fname_out = audio_example_path.joinpath(audio_name_tag).with_suffix('.wav')
    torchaudio.save(fname_out, wav_data, samplerate)


def save_specgram_fig(writer, global_step, audio_vec, spec_name_tag):
    # make sure wav is in range[-1, 1] before saving
    audio_vec /= audio_vec.abs().max()

    # dont print divide by zero warnings from spectrograms
    with np.errstate(divide='ignore'):

        fig = plt.figure()
        spec, freqs, times, im_ax = plt.specgram(audio_vec, NFFT=EXAMPLE_SPECTROGRAM_NFFT,
                                                    Fs=EXAMPLE_SAMPLING_FREQUENCY,
                                                    aspect='auto', vmin=-100, vmax=-20, cmap='coolwarm')
        plt.colorbar()
        writer.add_figure(spec_name_tag, fig, close=True, global_step=global_step)

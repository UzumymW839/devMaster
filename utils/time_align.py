import numpy as np
import scipy.signal as sig
import torch
import torchaudio

# TODO: integrate old test for this
# TODO: re-write this to be in torch

def time_align(x, y, resamp_fac=8, old_samplerate=16000):
    """
    Time delay compensation of x, based on y.
    The output signal is x, aligned with y based on autocorrelation.

    This function assumes the input is torch tensors with shape
    (channels, samples) and requires preprocessing by function prep_sig().

    The resampling factor resamp_fac determines the amount of
    oversampling done before compensating the delay in the oversampled signals
    (for a finer compensation).
    """
    x_hi = prep_sig(x, resamp_fac=resamp_fac, old_samplerate=old_samplerate)
    y_hi = prep_sig(y, resamp_fac=resamp_fac, old_samplerate=old_samplerate)

    sign_comp, lag_comp = determine_delay_and_sign(x_hi, y_hi)

    # compensate the lag in oversampled time-domain
    x_hi_comp = torch.roll(x_hi, shifts=-lag_comp, dims=1)

    # lag compensation check
    #corr_seq_2 = sig.correlate(in1=x_hi_comp, in2=y_hi, mode='full')
    #lag_2 = np.argmax(np.abs(corr_seq_2))
    #lag_comp_2 = lag_2 -(corr_seq_2.shape[0]-1)//2
    #print(lag_comp_2)

    # sign compensation as well
    x_hi_comp = sign_comp * x_hi_comp

    return unprep_sig(x_hi_comp, resamp_fac=resamp_fac, old_samplerate=old_samplerate)


def determine_delay_and_sign(x_hi, y_hi):
    """
    Determine the time delay in samples between two oversampled
    signals given the resampling factor.
    """
    corr_seq = torch.Tensor(sig.correlate(in1=x_hi.flatten(), in2=y_hi.flatten(), mode='full'))

    #corr_seq_2 = torch.nn.functional.conv1d(input=x_hi[None, ...], weight=y_hi[None, ...], padding='same')
    #import pdb
    #pdb.set_trace()

    # find the highest (abs. valued) peak and its sign
    lag_idx = torch.argmax(torch.abs(corr_seq))
    sign_comp = torch.sign(corr_seq[lag_idx])

    # compute the lag in samples
    return sign_comp, int(lag_idx -(corr_seq.shape[0] - 1) // 2)


def prep_sig(x, resamp_fac, old_samplerate=16000):
    """
    Prepare the signal for fine time-alignment by moving to CPU,
    converting to numpy.ndarray and resampling.

    Input x assumed to have shape (1, time), e.g. (1, 16000).
    Output has shape (time).
    """
    return torchaudio.functional.resample(waveform=x,
                                          orig_freq=old_samplerate,
                                           new_freq=old_samplerate * resamp_fac)


def unprep_sig(x_hi, resamp_fac, old_samplerate=16000):
    """
    Undo the upsampling and conversion done in prep_sig.

    Input has shape (1, time).
    Output is tensor with shape (1, old_len) at old sample rate.
    """
    return torchaudio.functional.resample(waveform=x_hi,
                                       orig_freq=resamp_fac * old_samplerate,
                                       new_freq=old_samplerate)

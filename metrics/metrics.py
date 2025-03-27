"""
This module contains metrics for speech enhancement/intelligibility/
bandwidth extension/noise reduction etc.

All metrics should inherit BaseMetric, and for testing compatibility have the class properties
_perfect_score (maximum possible value for perfect input)
_improvement_higher_scores (bool whether an improvement in the metric leads to a higher score, set false for distances etc.)
"""

from abc import abstractmethod
import torch
from pesq import pesq
import pystoi
import numpy as np
from metrics.haspi.audiogram import AUDIOGRAM_REF_CLARITY, Listener
from metrics.mbstoi import mbstoi
from metrics.haspi import haspi_v2_be

DEFAULT_METRIC_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseMetric(torch.nn.Module):

    _perfect_score = np.nan
    _improvement_higher_scores = True

    def __init__(self, device=DEFAULT_METRIC_DEVICE) -> None:
        super().__init__()
        self.device = device

    @abstractmethod
    def forward(self, predicted, target):
        raise NotImplementedError()


class STFTMagSq(torch.nn.Module):
    """
    STFT squared Magnitude computation using torch, on given device
    """
    def __init__(self, nfft: int=512, hop:int=256, device=DEFAULT_METRIC_DEVICE):
        super().__init__()

        self.nfft = nfft
        self.hop = hop
        self.window = torch.hann_window(nfft, device=device)


    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the STFT magnitude of x.
        Input is assumed to have shape (batch, time)
        """
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          return_complex=True)

        # after stft, shape is (batch, freq, OLA segments)
        mag = torch.abs(stft)

        # (now batch, freq, OLA segments)
        return mag.square()


class SNRMetric(BaseMetric):

    _perfect_score = torch.inf
    _improvement_higher_scores = True
    __name__ = 'SNR'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        This function computes the time-domain SNR.
        Here, noise is defined as all deviation from target.

        Inputs are assumed to come in as (batch, time).
        """
        s2_sum = torch.sum(target.square())
        e2_sum = torch.sum((predicted - target).square())

        # compute ratio without dividing by zero
        snr_lin = s2_sum / e2_sum# .clamp(min=1e-8)

        # use .item() here in order to match the template
        return 10*torch.log10(snr_lin).item()


class LSDMetric(BaseMetric):

    _perfect_score = 0.0
    _improvement_higher_scores = False
    __name__ = 'LSD'

    def __init__(self, log10_min_val: float=1e-8, **kwargs) -> None:
        super().__init__(**kwargs)
        # create object for computation of LSD metric
        self.stftmagsq = STFTMagSq(nfft=2048, hop=512, device=self.device)
        self.log10_min_val = log10_min_val

    def safe_log10(self, x: torch.Tensor) -> torch.Tensor:
        """
        clamp -> set values below min_val to this value, to avoid log(0)
        """
        return torch.log10(x.clamp(min=self.log10_min_val))

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Log spectral distance between two signals.
        Assumes inputs of shape (batch, time)
        """
        specs_p = self.safe_log10(self.stftmagsq(predicted))
        specs_t = self.safe_log10(self.stftmagsq(target)) # (batch, freq, OLA segments)

        # RMS over frequency -> (batch, OLA segments)
        lsd_per_seg = (specs_p - specs_t).square().mean(dim=1).sqrt()

        # average over segments, then over batch
        return lsd_per_seg.mean(dim=-1).mean().item()


class PESQMetric(BaseMetric):
    """
    Compute the PESQ measure.
    Inputs are assumed to come in as (batch, time).

    fs_desired is the sampling rate
    mode is either
    - wb for wideband
    - nb for narrowband
    """

    _perfect_score = 4.64
    _improvement_higher_scores = True
    __name__ = 'PESQ'

    def __init__(self, samplerate: int=16000, mode: str='wb', **kwargs) -> None:
        super().__init__(**kwargs)
        self.samplerate = samplerate
        self.mode = mode

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Inputs are assumed to come in as (batch, time).
        """
        ref = target.view(-1).detach().cpu().numpy()
        deg = predicted.view(-1).detach().cpu().numpy()

        try:
            score = pesq(fs=self.samplerate,
                        ref=ref, deg=deg,
                        mode=self.mode)
        except Exception as e:
            print("Error in processing files with PESQ: ")
            print(e)
            print("setting output value to NaN for now...")
            score = np.nan

        return score


class STOIMetric(BaseMetric):
    """
    Computes the STOI (short time objective intelligibility) measure
    which correlates with speech intelligibility.

    extended refers to using the eSTOI variant i believe.
    https://pypi.org/project/pystoi/
    """

    _perfect_score = 1.00
    _improvement_higher_scores = True
    __name__ = 'STOI'

    def __init__(self, samplerate: int=16000, extended: bool=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.samplerate = samplerate
        self.extended = extended

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        clean = target.view(-1).detach().cpu().numpy()
        denoised = predicted.view(-1).detach().cpu().numpy()

        score = pystoi.stoi(clean, denoised, self.samplerate, extended=self.extended)

        return score


class SISDRMetric(BaseMetric):
    """
    Compute the scale-invariant (SI) speech to distortion ratio (SDR).
    SI-SDR = 10*log10( ||z||_2^2 / ||z - s_hat||_2^2),
    where z = (s_hat^T * s) / (||s||_2^2) * s.
    """

    _perfect_score = 100
    _improvement_higher_scores = True
    __name__ = 'SISDR'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def pow_np_norm(self, x: np.ndarray) -> np.ndarray:
        """
        Compute L2 norm using numpy.
        y = || x ||_2^2
        """
        return np.square(np.linalg.norm(x, ord=2))

    def pow_norm(self, s1: np.ndarray, s2: np.ndarray):
        """
        Intermediate computation for SI-SDR.
        y = s1^T * s2,
        where * denotes the inner product.
        """
        return np.sum(s1 * s2)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> float:

        # convert to numpy array
        target = target.view(-1).detach().cpu().numpy()
        predicted = predicted.view(-1).detach().cpu().numpy()

        target = self.pow_norm(predicted, target) * target / self.pow_np_norm(target)
        noise = predicted - target
        return 10 * np.log10(self.pow_np_norm(target) / self.pow_np_norm(noise))


class MBSTOIMetric(BaseMetric):
    """
    Compute the MBSTOI measure.
    Inputs are assumed to come in as (channel, time).

    fs_desired is the sampling rate
    """

    _perfect_score = 1.0
    _improvement_higher_scores = True
    __name__ = 'MBSTOI'

    def __init__(self, samplerate: int=16000, **kwargs) -> None:
        super().__init__(**kwargs)
        self.samplerate = samplerate


    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Inputs are assumed to come in as (channel, time).
        """

        ref = target.detach().cpu().numpy()
        deg = predicted.detach().cpu().numpy()

        score = mbstoi(
            left_ear_clean=ref[0,:],
            right_ear_clean=ref[1,:],
            left_ear_noisy=deg[0,:],
            right_ear_noisy=deg[1,:],
            sr_signal=self.samplerate
        )

        return score


class HASPIMetric(BaseMetric):
    """
    Compute the HASPI measure.
    Inputs are assumed to come in as (channel, time).

    fs_desired is the sampling rate
    """

    _perfect_score = 1.0
    _improvement_higher_scores = True
    __name__ = 'HASPI'

    def __init__(self, samplerate: int=16000, **kwargs) -> None:
        super().__init__(**kwargs)
        self.samplerate = samplerate
        self.listener = Listener(audiogram_left=AUDIOGRAM_REF_CLARITY,
                                 audiogram_right=AUDIOGRAM_REF_CLARITY)


    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Inputs are assumed to come in as (channel, time).
        """
        ref = target.detach().cpu().numpy()
        deg = predicted.detach().cpu().numpy()

        score = haspi_v2_be(
            reference_left=ref[0,:],
            reference_right=ref[1,:],
            processed_left=deg[0,:],
            processed_right=deg[1,:],
            sample_rate=self.samplerate,
            listener=self.listener
        )

        return score


if __name__ == '__main__':
    x = torch.randn(2, 4*16000)
    y = x + 0.01 * torch.randn(2,4*16000)

    mbstoi_met = MBSTOIMetric()
    haspi_met = HASPIMetric()
    mbstoi_val = mbstoi_met(predicted=y, target=x)
    haspi_val = haspi_met(predicted=y, target=x)

    print(mbstoi_val, haspi_val)

import torch



class Normalization(torch.nn.Module):
    """
    This transform computes statistics for mean-variance normalization
    On each channel of the noisy input data.

    The target data (from the same reference channel!) is normalized by the same factor
    as the noisy data in the reference channel.

    reference_channel_target: the channel of the noisy data on which statistics
        the target data is supposed to be normalized by.
    """

    def __init__(self, reference_channel_target=0):
        super().__init__()
        self.reference_channel_target = reference_channel_target

    def forward(self, noisy_data, target_data):
        # inputs assumed to have shape (channels, time) and are normalized for each channel individually

        # compute mean and standard deviation over channels of noisy audio
        noisy_mean = torch.mean(noisy_data, dim=1)
        noisy_std = torch.std(noisy_data, dim=1)

        #noisy_std = torch.clamp(noisy_std, min=1e-16)

        normalized_noisy_data = (noisy_data - noisy_mean[:, None]) / noisy_std[:, None]
        normalized_target_data = (target_data - noisy_mean[self.reference_channel_target, None]) / noisy_std[self.reference_channel_target, None]

        return normalized_noisy_data, normalized_target_data


class MaxNorm(Normalization):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, noisy_data, target_data):
        max_noisy = torch.amax(torch.abs(noisy_data), dim=1)

        normalized_noisy_data = noisy_data / max_noisy[:, None]
        normalized_target_data = target_data / max_noisy[self.reference_channel_target, None]

        return normalized_noisy_data, normalized_target_data


if __name__ == '__main__':

    from utils.audio_utils import torch_rms

    noisy = torch.randn(2, 16000)
    target = 0.5 * torch.randn(2,16000)
    #target[1,:] *= 20
    noisy[1,:] *= 0.1

    norm = Normalization(reference_channel_target=[0,1])

    n_noisy, n_target = norm(noisy, target)

    print(torch_rms(n_noisy, axis=1))
    print(torch_rms(n_target, axis=1))

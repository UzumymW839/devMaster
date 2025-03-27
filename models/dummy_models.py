import torch
import torchaudio
from models.base_model import BaseModel

#device = torch.device('cpu')
STANDARD_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class NoisyInputDummy(BaseModel):
    def __init__(self, configuration=None, selected_channels=[0]) -> None:
        super().__init__(configuration)
        self.selected_channels = selected_channels

    def forward(self, x):
        # expected input shape: (batch, channels, time)
        return x[:, self.selected_channels, :]

    def flatten_all_params(self):
        pass


class NoisySTFTDummy(NoisyInputDummy):
    def __init__(self,
                 n_fft=512,
                 window_device=STANDARD_DEVICE,
                 **kwargs
                 ) -> None:

        super().__init__(**kwargs)

        hop_length = n_fft // 2

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None,
            window_fn=torch.hann_window,
            wkwargs={'device': window_device}
        )

        self.inverse_spec_transform = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            window_fn=torch.hann_window,
            wkwargs={'device': window_device}
        )

    def forward(self, x: torch.Tensor):
        # expected input shape: (batch, channels, time)
        # channel selection
        x = x[:, self.selected_channels, :]

        time_length = x.shape[-1]

        x_stft = self.spec_transform(x) # -> (batch, channel, freq, time ???)

        return self.inverse_spec_transform(x_stft, length=time_length)


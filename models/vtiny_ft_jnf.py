
from typing_extensions import Literal
import torch

import torchaudio
from models.base_model import BaseModel


#device = torch.device('cpu')
STANDARD_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class vTiny_FT_JNF(BaseModel):
    """
    Mask estimation network composed of two LSTM layers. One LSTM layer uses the frequency-dimension as sequence input
    and the other LSTM uses the time-dimension as input.
    """
    def __init__(self,
                 #n_time_steps: int,
                 n_freqs: int = 257,
                 n_channels: int = 2,
                 n_lstm_hidden1: int = 128,
                 n_lstm_hidden2: int = 32,
                 bidirectional: bool = False,
                 freq_first: bool = True,
                 output_type: Literal['IRM', 'CRM', 'Nch_CRM'] = 'CRM',
                 output_activation: Literal['sigmoid', 'tanh', 'linear'] = 'tanh',
                 dropout: float = 0,
                 append_freq_idx: bool = False,
                 permute_freqs: bool = False,
                 channels_to_filter: list = [1],
                 use_ch: list = [0, 1],
                 with_regressor: bool = False,
                 window_device=STANDARD_DEVICE):
        """
        Initialize model.
        # :param n_time_steps: number of STFT time frames in the input signal
        :param n_freqs: number of STFT frequency bins in the input signal
        :param n_channels: number of channel in the input signal
        :param n_lstm_hidden1: number of LSTM units in the first LSTM layer
        :param n_lstm_hidden2: number of LSTM units in the second LSTM layer
        :param bidirectional: set to True for a bidirectional LSTM
        :param freq_first: process frequency dimension first if freq_first else process time dimension first
        :param output_type: output_type: set to 'IRM' for real-valued ideal ratio mask (IRM) and to 'CRM' for complex IRM
        :param output_activation: the activation function applied to the network output (options: 'sigmoid', 'tanh', 'linear')
        :param dropout: dropout percentage (default: no dropout)
        """
        super(vTiny_FT_JNF, self).__init__()

        #self.n_time_steps = n_time_steps
        self.n_freqs = n_freqs
        self.n_channels = n_channels
        self.n_lstm_hidden1 = n_lstm_hidden1
        self.n_lstm_hidden2 = n_lstm_hidden2
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_type = output_type
        self.output_activation = output_activation
        self.freq_first = freq_first
        self.append_freq_idx = append_freq_idx
        self.permute = permute_freqs
        self.with_regressor = with_regressor

        lstm_input = 2*n_channels
        if self.append_freq_idx:
            lstm_input += 1

        self.lstm1 = torch.nn.LSTM(input_size=lstm_input, hidden_size=self.n_lstm_hidden1, bidirectional=bidirectional, batch_first=False)
        self.lstm1.flatten_parameters()

        self.lstm1_out = 2*self.n_lstm_hidden1 if self.bidirectional else self.n_lstm_hidden1
        lstm2_input = self.lstm1_out
        if self.append_freq_idx:
            lstm2_input+= 1

        self.lstm2 = torch.nn.LSTM(input_size=lstm2_input, hidden_size=self.n_lstm_hidden2, bidirectional=bidirectional, batch_first=False)
        self.lstm2.flatten_parameters()
        self.lstm2_out = 2*self.n_lstm_hidden2 if self.bidirectional else self.n_lstm_hidden2

        self.dropout = torch.nn.Dropout(p=self.dropout)

        if self.output_type == 'IRM':
            self.linear_out_features = 1
        elif self.output_type == 'CRM':
            self.linear_out_features = 2
        elif self.output_type == 'Nch_CRM':
            self.linear_out_features = self.n_channels * 2 # 2 complex ratio masks a 2 features each
        else:
            raise ValueError(f'The output type {output_type} is not supported.')
        self.ff = torch.nn.Linear(self.lstm2_out, out_features=self.linear_out_features)

        if self.output_activation == 'sigmoid':
            self.mask_activation = torch.nn.Sigmoid()
        elif self.output_activation == 'tanh':
            self.mask_activation = torch.nn.Tanh()
        elif self.output_activation == 'linear':
            self.mask_activation = torch.nn.Identity()

        n_fft = 2*(self.n_freqs-1)
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

        self.channels_to_filter = channels_to_filter
        self.use_ch = use_ch

        if self.with_regressor:
            self.regressor_lstm1 = torch.nn.Conv2d(in_channels=self.n_lstm_hidden1,
                                                out_channels=512,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)
            self.regressor_lstm2 = torch.nn.Conv2d(in_channels=n_lstm_hidden2,
                                                out_channels=128,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)


    def flatten_all_params(self):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()


    def forward(self, x: torch.Tensor, return_mask: bool=False):
        self.flatten_all_params()

        x = x[:, self.use_ch, :]

        time_length = x.shape[-1]
        # input: batch, channel, time
        x = self.spec_transform(x) # -> (batch, channel, freq, time ???)

        x_r = torch.real(x)
        x_i = torch.imag(x)

        x = torch.cat([x_r, x_i], dim=1) # (batch, channel x re/im, freq, time)
        """
        Implements the forward pass of the model.
        :param x: input with shape [BATCH, CHANNEL, FREQ, TIME]
        :return: the output mask [BATCH, 1 (IRM) or 2 (CRM) , FREQ, TIME]
        """
        n_batch, n_channel, n_freq, n_times = x.shape

        if not self.freq_first: # narrow_band
            seq_len = n_times
            tmp_batch = n_batch*n_freq
            x = x.permute(3,0,2,1).reshape(n_times, n_batch*n_freq, n_channel)
        else: # wide_band
            seq_len = n_freq
            tmp_batch = n_batch*n_times
            x = x.permute(2,0,3,1).reshape(n_freq, n_batch*n_times, n_channel)

        if self.permute:
            perm = torch.randperm(seq_len)
            inv_perm = torch.zeros(seq_len, dtype=int)
            for i, val in enumerate(perm):
                inv_perm[val] = i
            x = x[perm]
        else:
            perm = torch.arange(seq_len)

        if self.append_freq_idx:
            if not self.freq_first: # narrow_band:
                freq_bins = torch.arange(n_freq).repeat(n_batch*n_times).reshape(seq_len, tmp_batch, 1).to(x.device)
                x = torch.concat((x, freq_bins), dim=-1)
            else: # wide_band
                freq_bins = torch.arange(n_freq).repeat(int(seq_len/n_freq))[perm]
                freq_bins = freq_bins.unsqueeze(1).unsqueeze(1).broadcast_to((seq_len, tmp_batch, 1)).to(x.device)
                x = torch.concat((x, freq_bins), dim=-1)

        x, _ = self.lstm1(x)
        if self.with_regressor:
            self.lstm1_out = self.regressor_lstm1(x)
        x = self.dropout(x)

        if self.permute:
            x = x[inv_perm]

        if not self.freq_first: # narrow_band -> wide_band
            seq_len = n_freq
            tmp_batch = n_batch*n_times
            x = x.reshape(n_times, n_batch, n_freq, self.lstm1_out).permute(2,1,0,3).reshape(n_freq, n_batch*n_times, self.lstm1_out)
        else: # wide_band -> narrow_band
            seq_len = n_times
            tmp_batch = n_batch*n_freq
            x =  x.reshape(n_freq, n_batch, n_times, self.lstm1_out).permute(2,1,0,3).reshape(n_times, n_batch*n_freq, self.lstm1_out)

        if self.permute:
            perm = torch.randperm(seq_len)
            inv_perm = torch.zeros(seq_len, dtype=int)
            for i, val in enumerate(perm):
                inv_perm[val] = i
            x = x[perm]
        else:
            perm = torch.arange(seq_len)

        if self.append_freq_idx:
            if self.freq_first: # wide_band
                freq_bins = torch.arange(n_freq).repeat(n_batch*n_times).reshape(seq_len, tmp_batch, 1).to(x.device)
                x = torch.concat((x, freq_bins), dim=-1)
            else: # narrow_band
                freq_bins = torch.arange(n_freq).repeat(int(seq_len/n_freq))[perm]
                freq_bins = freq_bins.unsqueeze(1).unsqueeze(1).broadcast_to((seq_len, tmp_batch, 1)).to(x.device)
                x = torch.concat((x, freq_bins), dim=-1)

        x, _ = self.lstm2(x)
        if self.with_regressor:
            self.lstm2_out = self.regressor_lstm2(x)
        x = self.dropout(x)

        if self.permute:
            x = x[inv_perm]

        x = self.ff(x)

        if not self.freq_first: # wide_band -> input shape
            x = x.reshape(n_freq, n_batch, n_times, self.linear_out_features).permute(1,3,0,2)
        else: # narrow_band -> input shape
            x = x.reshape(n_times, n_batch, n_freq, self.linear_out_features).permute(1,3,2,0)

        m = self.mask_activation(x) # mask shape: (batch, ch*re/im=4, n_freq=257, n_times=188)

        complex_x = x_r + 1j * x_i # before and after adding: (batch, ch, freq, time)

        if self.output_type == 'Nch_CRM':

            real_idx = [0 + 2*x for x in range(self.n_channels)]
            imag_idx = [1 + 2*x for x in range(self.n_channels)]

            # combine real and imaginary channels for 2ch complex masks
            complex_m = m[:,real_idx, ...] + 1j * m[:,imag_idx, ...]

            # shape after adding: (batch, ch, freq, time)

            # filtering individual channels with corresponding masks
            y = complex_m * complex_x[:, self.channels_to_filter, ...]

            # sum up channels to single output channel signal
            y = torch.sum(y, dim=1, keepdim=True)

        else:
            complex_m = m[:,[0], ...] + 1j * m[:,[1], ...]
            y = complex_m * complex_x[:, self.channels_to_filter, ...]

        if return_mask:
            #print(f'complex m shape: {complex_m.shape}')
            #print(complex_m.dtype)
            #print(f'max val: {torch.amax(torch.abs(complex_m))}, min val: {torch.amin(torch.abs(complex_m))}')
            return self.inverse_spec_transform(y, length=time_length), complex_m
            #return complex_m
        
        if self.with_regressor:
            return self.inverse_spec_transform(y, length=time_length), self.lstm1_out, self.lstm2_out

        return self.inverse_spec_transform(y, length=time_length)


if __name__ == '__main__':
    n_ch = 3

    x = torch.randn(4, 3, 3*16000).to('cuda')
    #mod = FT_JNF().to('cuda')

    #mod = FT_JNF(n_channels=1, channels_to_filter=[0], use_ch=[0]).to('cuda')

    mod = vTiny_FT_JNF(output_type='Nch_CRM', n_channels=3, channels_to_filter=[0, 1, 2],
                 use_ch=[0, 1, 2]).to('cuda')
    y = mod(x)
    print(y.shape)
    # all params (probably includes STFT and so on)
    #pytorch_total_params = sum(p.numel() for p in mod.parameters())
    #pytorch_total_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    #print(pytorch_total_params)

    """
    import pdb

    mod.to(device='cuda:0')
    z = x[:,[1],:].clone().to(device='cuda:0')
    x = x.to(device='cuda:0')
    print(x.device)
    pdb.set_trace()
    y = mod(x)
    pdb.set_trace()
    """

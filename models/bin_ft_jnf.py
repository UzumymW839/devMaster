
from typing_extensions import Literal
import torch

from models.ft_jnf import FT_JNF



class Bin_FT_JNF(FT_JNF):
    """
    Same as FT_JNF, but with Binaural output
    """
    def __init__(self,
                 output_type: Literal['CRM'] = 'CRM',
                 channels_to_filter: list = [[0, 1], [0,1]], # left and right!
                 use_ch: list = [0, 1],
                 **kwargs):

        super().__init__(**kwargs)


        if self.output_type == 'CRM':
            self.linear_out_features = 2 * 2 * 2 # real+imag (2) for each input channel (2) and each output channel (2)
        else:
            raise ValueError(f'The output type {output_type} is not supported.')
        self.ff = torch.nn.Linear(self.lstm2_out, out_features=self.linear_out_features)

        self.channels_to_filter = channels_to_filter
        self.use_ch = use_ch


    def forward(self, x: torch.Tensor):
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
        x = self.dropout(x)

        if self.permute:
            x = x[inv_perm]

        x = self.ff(x)

        if not self.freq_first: # wide_band -> input shape
            x = x.reshape(n_freq, n_batch, n_times, self.linear_out_features).permute(1,3,0,2)
        else: # narrow_band -> input shape
            x = x.reshape(n_times, n_batch, n_freq, self.linear_out_features).permute(1,3,2,0)

        m = self.mask_activation(x) # mask shape: (batch, (2 in ch)*(2 out ch) * (re+im)=8, n_freq=257, n_times=188)

        m_r = m[:, [0,2,4,6], ...]
        m_i = m[:, [1,3,5,7], ...]

        complex_m = m_r + 1j * m_i # combine real and imaginary channels for 2ch complex masks
        # shape after adding: (batch, 4 ch, freq, time)

        # before and after adding: (batch, 4 ch, freq, time)
        complex_x = x_r + 1j * x_i

        y_left = torch.sum(
            complex_m[:, [0, 1], ...] * complex_x[:, self.channels_to_filter[0], ...],
            axis=1, keepdim=True)


        y_right = torch.sum(
            complex_m[:, [2, 3], ...] * complex_x[:, self.channels_to_filter[1], ...],
            axis=1, keepdim=True)

        y = torch.concat([y_left, y_right], dim=1)


        return self.inverse_spec_transform(y, length=time_length)


if __name__ == '__main__':
    x = torch.randn(4, 2, 3*16000).to('cuda')

    #mod = Bin_FT_JNF(n_channels=2, channels_to_filter=[[0, 1], [0,1]], use_ch=[0,1]).to('cuda')
    mod = Bin_FT_JNF(n_channels=2,
                     channels_to_filter=[[0, 1], [0,1]],
                     use_ch=[0,1],
                     n_freqs=129 #n_freqs=257
                     ).to('cuda')

    y = mod(x)
    print(y.shape)
    pytorch_total_params = sum(p.numel() for p in mod.parameters())
    print(pytorch_total_params)


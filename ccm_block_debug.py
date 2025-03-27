

import torch

class CMMBlock(torch.nn.Module):
    def __init__(self, m, n, c) -> None:
        super().__init__()

        self.m = m
        self.n = n
        self.c = c
        self.kernel_shape = (m+1, 2*n+1)
        self.basis_ndim = 3

        self.rotate_vector = torch.ones((self.basis_ndim,), dtype=torch.cfloat)
        self.rotate_vector[1] = -0.5 + 1j * torch.sqrt(torch.tensor(3))/2
        self.rotate_vector[2] = -0.5 - 1j * torch.sqrt(torch.tensor(3))/2

    def forward(self, x, X_mic):
        # inputs:
        #   x - the DNN output so far,
        #   X_mic - the complex-valued input mic. STFT
        #
        #   x expected shape: (channels, time, frequency) (real-valued)
        #   X_mic expected shape: (time, frequency)

        # TODO:channels should be integer-divisible by 3?

        # TODO: is X_mic in the paper the same as X?

        # TODO: CCM block is applied causally -> conv over time should not be centered around 0

        ### FIRST STAGE ###

        # reshape X to X' -> should have shape (3, channels/3, t, f) (both real-valued)
        # then casted as complex float
        x_dash = x.reshape(self.basis_ndim, self.c//self.basis_ndim, x.shape[1], x.shape[2]).cfloat()

        # create mask -> shape (channels/3, t, f) (now complex-valued)
        # v * X' is inner product
        H = torch.tensordot(self.rotate_vector, x_dash, dims=[[0], [0]])

        ### SECOND STAGE ###
        M = H.reshape(self.kernel_shape[0], self.kernel_shape[1],
                      self.c//(self.basis_ndim*self.kernel_shape[0]*self.kernel_shape[1]),
                      x.shape[1], x.shape[2]
                      ).squeeze() # as in paper
        #print(M.shape) # (filter_w, filter_h, t, f)

        #M = H.reshape(
        #    self.c//(self.basis_ndim*self.kernel_shape[0]*self.kernel_shape[1]), # out_channels=1
        #    x.shape[1], x.shape[2],
        #    self.kernel_shape[0], self.kernel_shape[1], # width, height
        #).squeeze() # remove dim0 = 1

        #print(M.shape) # (t, f, filter_w, filter_h)

        # reshape input and kernel to work with conv2d
        conv_input = X_mic #X_mic[None, None, ...] # (shape: batch=1, in_ch=1, t, f)
        conv_kernel = M # shape: (out_ch, in_ch/groups=1, t, f)

        # split input and kernel into real and imag parts (old torch version ugh!)
        #in_re = torch.real(conv_input)
        #in_im = torch.imag(conv_input)
        #m_re = torch.real(conv_kernel)
        #m_im = torch.imag(conv_kernel)

        # clean spectrum estimation
        #Y_re = torch.nn.functional.conv2d(input=in_re, weight=m_re, padding='same')
        #Y_im = torch.nn.functional.conv2d(input=in_im, weight=m_im, padding='same')
        #Y = Y_re + 1j * Y_im

        Y = self.complex_filter_operation(conv_input, conv_kernel)

        return Y#.squeeze()


    def complex_filter_operation(self, conv_input, conv_kernel):
        Y = torch.zeros_like(conv_input)
        #for i in range(-self.m, 0):
        #    for j in range(-self.n, self.n):
        for i in range(self.m+1):
            for j in range(2*self.n+1):
                Y += conv_input * conv_kernel[i, j, :, :]

        return Y


if __name__ == '__main__':
    c = 45
    t = 32
    f = 129
    x_dnn = torch.randn(c, t, f)
    X_mic = torch.randn(t,f) + 1j * torch.randn(t,f)

    m = 2
    n = 2
    check_val = 3*(m+1)*(2*n+1)
    print(check_val)
    # pretty hacky; i guess c should be determined automatically or like in W.Macks paper
    assert c == check_val


    cmmb = CMMBlock(m=n, n=n, c=c)

    Y = cmmb(x_dnn, X_mic)

    print(Y.shape, Y.dtype)
    print([t,f])
    assert Y.shape == [t, f]

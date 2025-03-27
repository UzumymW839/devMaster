"""
This module implements loss functions to be used in DNN training.
All classes should inherit from BaseLoss.
"""
from abc import abstractmethod
import torch
import torchaudio

DEFAULT_LOSS_DEVICE = torch.device('cpu')
#DEFAULT_LOSS_DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
STANDARD_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, predicted, target):
        raise NotImplementedError


class STFTMagLoss(BaseLoss):
    """
    STFT squared Magnitude loss computation using torch.
    """
    _perfect_score = 0.0

    def __init__(self, nfft=512, hop=256, reference_channel=0, device=DEFAULT_LOSS_DEVICE):
        super().__init__()

        self.nfft = nfft
        self.hop = hop
        self.window = torch.hann_window(nfft, device=device)
        self.reference_channel = reference_channel
        self.eps = 1e-8


    def compute_stft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the STFT magnitude of x.
        Input is assumed to have shape (batch, channels, time)
        """
        stft = torch.stft(x[:, self.reference_channel, :],
                          self.nfft,
                          self.hop,
                          window=self.window,
                          return_complex=True)

        # after stft, shape is (batch, freq, OLA segments)
        mag = torch.abs(stft)

        # clamp small values
        mag = torch.clamp(mag, min=self.eps)

        # (now batch, freq, OLA segments)
        return mag

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_mag = self.compute_stft_magnitude(predicted)
        target_mag = self.compute_stft_magnitude(target)

        return torch.nn.functional.l1_loss(input=predicted_mag, target=target_mag)


class WavSTFTMagLoss(STFTMagLoss):
    """
    STFT squared complex-valued loss computation using torch.
    """
    _perfect_score = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_mag = self.compute_stft_magnitude(predicted)
        target_mag = self.compute_stft_magnitude(target)

        wav_loss = torch.nn.functional.l1_loss(input=predicted, target=target)
        mag_loss = torch.nn.functional.l1_loss(input=predicted_mag, target=target_mag)

        return wav_loss + mag_loss
    
class KDLayerLoss(BaseLoss):
    """ Loss function for knowledge distillation between two complex masks"""
    def __init__(self, device=DEFAULT_LOSS_DEVICE):
        super().__init__()

    def forward(self,
                ce_loss: torch.Tensor,
                alpha: float,
                student_layer: torch.Tensor, 
                teacher_layer: torch.Tensor) -> torch.Tensor:
        return  alpha * ce_loss + (1-alpha) * torch.nn.functional.l1_loss(input=student_layer, target=teacher_layer)
    
class KDLSTMLayerLoss(BaseLoss):
    """Loss function for knwoledge distillation between two LSTM layer outputs with regressor"""
    def __init__(self, device=DEFAULT_LOSS_DEVICE):
        super().__init__()
        self.device = device
        self.soft_loss = torch.nn.L1Loss()

    def forward(self,
                ce_loss: torch.Tensor,
                alpha: float,
                teacher_layer: torch.Tensor,
                student_layer: torch.Tensor) -> torch.Tensor:
        l_loss = self.soft_loss(input=torch.mm(student_layer.mean(dim=0), student_layer.mean(dim=0).t()),
                                target=torch.mm(teacher_layer.mean(dim=0), teacher_layer.mean(dim=0).t()))
        return alpha * ce_loss + (1-alpha) * l_loss
    
class KDSelfSimilarityLoss(BaseLoss):
    """Loss function between the layers of two models using self similarity matirx"""
    def __init__(self, device=DEFAULT_LOSS_DEVICE):
        super().__init__()
        self.device = device
        print("Using KDSelfSimilarityLoss")
        self.soft_loss = torch.nn.L1Loss()
        #self.soft_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')

    def forward(self,
                ce_loss: torch.Tensor,
                alpha: float,
                teacher_LSTM1: torch.Tensor,
                teacher_LSTM2: torch.Tensor,
                teacher_linear: torch.Tensor,
                student_LSTM1: torch.Tensor,
                student_LSTM2: torch.Tensor,
                student_linear: torch.Tensor) -> torch.Tensor:
        # compute the self similarity matrix and the loss for LSTM1
        LSTM1_loss = self.soft_loss(input=torch.mm(student_LSTM1.mean(dim=0), student_LSTM1.mean(dim=0).t()), 
                                    target=torch.mm(teacher_LSTM1.mean(dim=0), teacher_LSTM1.mean(dim=0).t()))
                                    #torch.ones(student_LSTM1.size(0), device=self.device))

        # compute the self similarity matrix and the loss for LSTM2
        LSTM2_loss = self.soft_loss(input=torch.mm(student_LSTM2.mean(dim=0), student_LSTM2.mean(dim=0).t()),
                                    target=torch.mm(teacher_LSTM2.mean(dim=0), teacher_LSTM2.mean(dim=0).t()))
                                    #torch.ones(student_LSTM2.size(0), device=self.device))

        # compute the loss for the linear layer
        linear_loss = self.soft_loss(input=student_linear, 
                                     target=teacher_linear)
                                     #torch.ones(student_linear.size(0), device=self.device))

        # compute and return over all loss
        return alpha * ce_loss + (1-alpha) * (linear_loss + LSTM1_loss + LSTM2_loss)


if __name__ == '__main__':
    loss = KDSelfSimilarityLoss()

    tLSTM1 = torch.randn(514,502,512)
    sLSTM1 = torch.randn(514,502,64, requires_grad=True)
    tLSTM2 = torch.randn(502,514,128)
    sLSTM2 = torch.randn(502,514,16, requires_grad=True)
    tLinear = torch.randn(502,514,10)
    sLinear = torch.randn(502,514,10, requires_grad=True)

    l_loss = loss(ce_loss=torch.nn.functional.l1_loss(tLinear, sLinear),
              alpha=0.5,
              teacher_LSTM1=tLSTM1,
              teacher_LSTM2=tLSTM2,
              teacher_linear=tLinear,
              student_LSTM1=sLSTM1,
              student_LSTM2=sLSTM2,
              student_linear=sLinear)
    print(l_loss)

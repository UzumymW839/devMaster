import torch
import matplotlib.pyplot as plt

def plot_spectrogram(stft, title = "Spectrogram"):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(spectrogram, 
                      cmap="viridis", 
                      vmin=-100, 
                      vmax=0, 
                      origin="lower", 
                      aspect="auto")
    axis.set_title(title)
    plt.colorbar(img, ax=axis)
    plt.show()

def plot_mask(mask, title = "Mask"):
    mask = mask.numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(mask, 
                      cmap="viridis", 
                      origin="lower", 
                      aspect="auto")
    axis.set_title(title)
    plt.colorbar(img, ax=axis)
    plt.show()
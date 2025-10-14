import torch
from torch import nn


class Normalize1D(nn.Module):
    """
    Normalize spectrograms along the time dimension.
    Applies normalization: (x - mean) / std
    """
    def __init__(self, mean=0.0, std=1.0, save_masked_zeroes=True):
        """
        Args:
            mean: Mean value for normalization
            std: Standard deviation for normalization
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.save_masked_zeroes = save_masked_zeroes

    def forward(self, spectrogram):
        """
        Args:
            spectrogram: Input spectrogram tensor [batch, freq, time] or [batch, channels, freq, time]
        Returns:
            Normalized spectrogram
        """
        if self.save_masked_zeroes:
            zero_times = (spectrogram == 0).all(dim=-1, keepdim=True)
            
        spec_mean = spectrogram.mean(dim=-1, keepdim=True)
        spec_std = spectrogram.std(dim=-1, keepdim=True)
        spectrogram_norm = (spectrogram - spec_mean) / (spec_std + 1e-8)
        spectrogram_norm = spectrogram_norm * self.std + self.mean

        if self.save_masked_zeroes:
            spectrogram_norm = spectrogram_norm.masked_fill(zero_times, 0.0)

        return spectrogram_norm


class NormalizeGlobal(nn.Module):
    """
    Normalize spectrograms globally using all values in the spectrogram.
    """
    def __init__(self, eps=1e-8, save_zeroes=True):
        """
        Args:
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.eps = eps
        self.save_zeroes = save_zeroes

    def forward(self, spectrogram):
        """
        Args:
            spectrogram: Input spectrogram tensor [batch, freq, time] or [batch, channels, freq, time]
        Returns:
            Normalized spectrogram
        """
        if self.save_zeroes:
            zero_mask = (spectrogram == 0)

        if len(spectrogram.shape) == 3:
            # [batch, freq, time]
            mean = spectrogram.mean(dim=(1, 2), keepdim=True)
            std = spectrogram.std(dim=(1, 2), keepdim=True)
        else:
            # [batch, channels, freq, time]
            # Compute per-sample normalization
            mean = spectrogram.mean(dim=(1, 2, 3), keepdim=True)
            std = spectrogram.std(dim=(1, 2, 3), keepdim=True)
        
        spectrogram = (spectrogram - mean) / (std + self.eps)

        if self.save_zeroes:
            spectrogram = spectrogram.masked_fill(zero_mask, 0.0)

        return spectrogram

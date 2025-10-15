import torch
import random


class FrequencyMasking(torch.nn.Module):
    """
    Apply frequency masking to spectrograms for data augmentation.
    """
    def __init__(self, freq_mask_param=15, num_masks=1, p=0.5):
        """
        Args:
            freq_mask_param: Maximum number of consecutive frequency bands to mask
            num_masks: Number of frequency masks to apply
            p: Probability of applying the augmentation
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        self.p = p

    def forward(self, spectrogram):
        """
        Args:
            spectrogram: Input spectrogram tensor [channels, freq, time] or [batch, channels, freq, time]
        Returns:
            Masked spectrogram
        """
        if len(spectrogram.shape) == 3:
            return self.forward(spectrogram.unsqueeze(0)).squeeze(0)

        if self.p != 1.0 and torch.rand(1).item() > self.p:
            return spectrogram

        cloned = spectrogram.clone()

        num_mel_channels = cloned.shape[2]
            
        for _ in range(self.num_masks):
            f = random.randrange(0, self.freq_mask_param)
            f_zero = random.randrange(0, num_mel_channels - f)
            
            cloned[:, :, f_zero:f_zero + f, :] = 0
        
        return cloned


class TimeMasking(torch.nn.Module):
    """
    Apply time masking to spectrograms for data augmentation.
    """
    def __init__(self, time_mask_param=25, num_masks=1, p=0.5):
        """
        Args:
            time_mask_param: Maximum number of consecutive time steps to mask
            num_masks: Number of time masks to apply
            p: Probability of applying the augmentation
        """
        super().__init__()
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        self.p = p

    def forward(self, spectrogram):
        """
        Args:
            spectrogram: Input spectrogram tensor [channels, freq, time] or [batch, channels, freq, time]
        Returns:
            Masked spectrogram
        """
        if len(spectrogram.shape) == 3:
            return self.forward(spectrogram.unsqueeze(0)).squeeze(0)
        
        if self.p != 1.0 and torch.rand(1).item() > self.p:
            return spectrogram

        cloned = spectrogram.clone()

        len_spectro = cloned.shape[3]
            
        for _ in range(self.num_masks):
            t = random.randrange(0, self.time_mask_param)
            t_zero = random.randrange(0, len_spectro - t)
            
            cloned[:, :, :, t_zero:t_zero + t] = 0
        
        return cloned


class SpecAugment(torch.nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    """
    def __init__(
        self,
        freq_mask_param=15,
        time_mask_param=20,
        num_freq_masks=2,
        num_time_masks=2,
        p=1.0,
    ):
        super().__init__()
        self.freq_masking = FrequencyMasking(freq_mask_param, num_freq_masks, 1.0)
        self.time_masking = TimeMasking(time_mask_param, num_time_masks, 1.0)
        self.p = p

    def forward(self, spectrogram):
        """
        Apply SpecAugment to input spectrogram
        Args:
            spectrogram: [channels, freq, time] or [batch, channels, freq, time]
        """
        if self.p != 1.0 and torch.rand(1).item() > self.p:
            return spectrogram

        spectrogram = self.freq_masking(spectrogram)
        spectrogram = self.time_masking(spectrogram)
        return spectrogram

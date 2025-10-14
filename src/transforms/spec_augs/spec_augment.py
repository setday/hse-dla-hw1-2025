from torch import nn
import random


class FrequencyMasking(nn.Module):
    """
    Apply frequency masking to spectrograms for data augmentation.
    """
    def __init__(self, freq_mask_param=27, num_masks=1):
        """
        Args:
            freq_mask_param: Maximum number of consecutive frequency bands to mask
            num_masks: Number of frequency masks to apply
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks

    def forward(self, spectrogram):
        """
        Args:
            spectrogram: Input spectrogram tensor [channels, freq, time] or [batch, channels, freq, time]
        Returns:
            Masked spectrogram
        """
        cloned = spectrogram.clone()
        # Handle both 3D and 4D tensors
        if len(cloned.shape) == 3:
            # [channels, freq, time]
            num_mel_channels = cloned.shape[1]
            
            for _ in range(self.num_masks):
                f = random.randrange(0, self.freq_mask_param)
                f_zero = random.randrange(0, num_mel_channels - f)
                
                # Apply mask
                cloned[:, f_zero:f_zero + f, :] = 0
        else:
            # [batch, channels, freq, time]
            num_mel_channels = cloned.shape[2]
            
            for _ in range(self.num_masks):
                f = random.randrange(0, self.freq_mask_param)
                f_zero = random.randrange(0, num_mel_channels - f)
                
                # Apply mask
                cloned[:, :, f_zero:f_zero + f, :] = 0
        
        return cloned


class TimeMasking(nn.Module):
    """
    Apply time masking to spectrograms for data augmentation.
    """
    def __init__(self, time_mask_param=70, num_masks=1):
        """
        Args:
            time_mask_param: Maximum number of consecutive time steps to mask
            num_masks: Number of time masks to apply
        """
        super().__init__()
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def forward(self, spectrogram):
        """
        Args:
            spectrogram: Input spectrogram tensor [channels, freq, time] or [batch, channels, freq, time]
        Returns:
            Masked spectrogram
        """
        cloned = spectrogram.clone()
        # Handle both 3D and 4D tensors
        if len(cloned.shape) == 3:
            # [channels, freq, time]
            len_spectro = cloned.shape[2]
            
            for _ in range(self.num_masks):
                t = random.randrange(0, self.time_mask_param)
                t_zero = random.randrange(0, len_spectro - t)
                
                # Apply mask
                cloned[:, :, t_zero:t_zero + t] = 0
        else:
            # [batch, channels, freq, time]
            len_spectro = cloned.shape[3]
            
            for _ in range(self.num_masks):
                t = random.randrange(0, self.time_mask_param)
                t_zero = random.randrange(0, len_spectro - t)
                
                # Apply mask
                cloned[:, :, :, t_zero:t_zero + t] = 0
        
        return cloned


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    """
    def __init__(
        self,
        freq_mask_param=27,
        time_mask_param=70,
        num_freq_masks=2,
        num_time_masks=2
    ):
        super().__init__()
        self.freq_masking = FrequencyMasking(freq_mask_param, num_freq_masks)
        self.time_masking = TimeMasking(time_mask_param, num_time_masks)

    def forward(self, spectrogram):
        """
        Apply SpecAugment to input spectrogram
        Args:
            spectrogram: [channels, freq, time] or [batch, channels, freq, time]
        """
        spectrogram = self.freq_masking(spectrogram)
        spectrogram = self.time_masking(spectrogram)
        return spectrogram

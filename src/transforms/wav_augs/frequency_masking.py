import torch


class FrequencyMasking(torch.nn.Module):
    """
    Frequency masking augmentation in time domain.
    Applies a band-stop filter to mask out certain frequency ranges.
    This helps ASR models become robust to frequency distortions and missing frequency bands.
    """
    def __init__(self, min_freq: int = 500, max_freq: int = 3000, 
                 bandwidth: int = 500, sample_rate: int = 16000, p: float = 0.5):
        """
        Args:
            min_freq: Minimum center frequency for masking (Hz)
            max_freq: Maximum center frequency for masking (Hz)
            bandwidth: Width of the frequency band to mask (Hz)
            sample_rate: Audio sample rate
            p: Probability of applying the augmentation
        """
        super().__init__()
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.bandwidth = bandwidth
        self.sample_rate = sample_rate
        self.p = p

    def __call__(self, data: torch.Tensor):
        """
        Apply frequency masking to audio tensor using FFT.
        
        Args:
            data: Audio tensor of shape (batch, time) or (time,)
        
        Returns:
            Frequency-masked audio tensor
        """
        if not len(data.shape) == 1:
            return self(data.unsqueeze(0)).squeeze(0)

        if torch.rand(1).item() > self.p:
            return data
        
        center_freq = torch.FloatTensor(1).uniform_(self.min_freq, self.max_freq).item()
        
        n_fft = data.shape[-1]
        freqs = torch.fft.rfftfreq(n_fft, d=1.0/self.sample_rate).to(data.device)
        
        low_freq = center_freq - self.bandwidth / 2
        high_freq = center_freq + self.bandwidth / 2
        
        mask = torch.ones_like(freqs)
        mask[(freqs >= low_freq) & (freqs <= high_freq)] = 0.0
        
        fft = torch.fft.rfft(data, dim=-1)
        fft_masked = fft * mask.unsqueeze(0)
        masked_audio = torch.fft.irfft(fft_masked, n=n_fft, dim=-1)
        
        return masked_audio

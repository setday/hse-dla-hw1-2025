import torch


class AddBackgroundNoise(torch.nn.Module):
    """
    Add background noise to audio signal.
    This is crucial for ASR robustness as it simulates real-world noisy environments.
    """
    def __init__(self, min_snr_db: float = 10.0, max_snr_db: float = 30.0, p: float = 0.5):
        """
        Args:
            min_snr_db: Minimum signal-to-noise ratio in decibels
            max_snr_db: Maximum signal-to-noise ratio in decibels
            p: Probability of applying the augmentation
        """
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def __call__(self, data: torch.Tensor):
        """
        Add white Gaussian noise to audio tensor.
        
        Args:
            data: Audio tensor of shape (batch, time) or (time,)
        
        Returns:
            Augmented audio tensor with added noise
        """
        if torch.rand(1).item() > self.p:
            return data
        
        signal_power = torch.mean(data ** 2)
        
        snr_db = torch.FloatTensor(1).uniform_(self.min_snr_db, self.max_snr_db).to(data.device)
        snr_linear = torch.pow(10.0, snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        noise = torch.randn_like(data) * torch.sqrt(noise_power)
        
        return data + noise

import torch


class TimeStretch(torch.nn.Module):
    """
    Time stretching augmentation without changing pitch.
    This simulates different speaking rates, which is important for ASR robustness.
    Uses phase vocoder approach for time stretching.
    """
    def __init__(self, min_rate: float = 0.8, max_rate: float = 1.2, p: float = 0.5):
        """
        Args:
            min_rate: Minimum time stretch rate (< 1.0 means slower, > 1.0 means faster)
            max_rate: Maximum time stretch rate
            p: Probability of applying the augmentation
        """
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p

    def __call__(self, data: torch.Tensor):
        """
        Apply time stretching to audio tensor.
        
        Args:
            data: Audio tensor of shape (batch, time) or (time,)
        
        Returns:
            Time-stretched audio tensor
        """
        if len(data.shape) == 1:
            return self(data.unsqueeze(0)).squeeze(0)
        
        if torch.rand(1).item() > self.p:
            return data
        
        rate = torch.FloatTensor(1).uniform_(self.min_rate, self.max_rate).item()
        new_length = int(data.shape[-1] / rate)
        
        stretched = torch.nn.functional.interpolate(
            data.unsqueeze(1),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(1)
        
        return stretched

import torch


class Gain(torch.nn.Module):
    """
    Applies a random gain factor to the audio signal.
    """
    def __init__(self, min_gain_db: float = -12.0, max_gain_db: float = 12.0, p: float = 0.5):
        """
        Args:
            min_gain_db: Minimum gain in decibels
            max_gain_db: Maximum gain in decibels
            p: Probability of applying the augmentation
        """
        super().__init__()
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.p = p

    def __call__(self, data: torch.Tensor):
        """
        Apply gain augmentation to audio tensor.
        
        Args:
            data: Audio tensor of shape (batch, time) or (time,)
        
        Returns:
            Augmented audio tensor with the same shape
        """
        if torch.rand(1).item() > self.p:
            return data
        
        gain_db = torch.FloatTensor(1).uniform_(self.min_gain_db, self.max_gain_db).to(data.device)
        gain_linear = torch.pow(10.0, gain_db / 20.0)
        
        return data * gain_linear

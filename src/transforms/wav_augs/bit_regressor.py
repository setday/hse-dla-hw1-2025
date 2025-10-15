import torch


class BitRegression(torch.nn.Module):
    """
    Bit regression augmentation.
    This simulates different bit rates and quantization levels,
    which is important for ASR to be robust against compression artifacts.
    """
    def __init__(self, min_bits: int = 8, max_bits: int = 16, p: float = 0.5):
        """
        Args:
            min_bits: Minimum bit depth
            max_bits: Maximum bit depth
            p: Probability of applying the augmentation
        """
        super().__init__()
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.p = p

    def __call__(self, data: torch.Tensor):
        """
        Apply pitch shifting to audio tensor.
        
        Args:
            data: Audio tensor of shape (batch, time) or (time,)
        
        Returns:
            Pitch-shifted audio tensor
        """
        if len(data.shape) == 1:
            return self(data.unsqueeze(0)).squeeze(0)
        
        if torch.rand(1).item() > self.p:
            return data
        
        bits = torch.randint(self.min_bits, self.max_bits + 1, (1,)).item()
        quantization_levels = 2 ** bits
        data = (data * (quantization_levels - 1)).round().clamp(0, quantization_levels - 1) / (quantization_levels - 1)

        return data

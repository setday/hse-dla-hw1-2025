import torch
from torch_audiomentations import PitchShift as TorchPitchShift


class PitchShift(torch.nn.Module):
    """
    Pitch shifting augmentation.
    This simulates different pitch levels,
    which is important for ASR to be robust against variations in speaker pitch.
    """
    def __init__(self, min_semitones: int = -4, max_semitones: int = 4, p: float = 0.5):
        """
        Args:
            min_semitones: Minimum number of semitones to shift
            max_semitones: Maximum number of semitones to shift
            p: Probability of applying the augmentation
        """
        super().__init__()
        self.augmenter = TorchPitchShift(min_transpose_semitones=min_semitones, max_transpose_semitones=max_semitones, p=p, sample_rate=16000)

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
        
        data = data.unsqueeze(1)
        pitched_data = self.augmenter(data)
        pitched_data = pitched_data.squeeze(1)

        return pitched_data
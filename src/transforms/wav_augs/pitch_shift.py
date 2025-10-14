import torch


class PitchShift(torch.nn.Module):
    """
    Pitch shifting augmentation.
    This simulates different voice characteristics (male/female, age variations),
    which is important for ASR to be speaker-independent.
    """
    def __init__(self, min_semitones: float = -4.0, max_semitones: float = 4.0, 
                 sample_rate: int = 16000, p: float = 0.5):
        """
        Args:
            min_semitones: Minimum pitch shift in semitones (negative = lower pitch)
            max_semitones: Maximum pitch shift in semitones (positive = higher pitch)
            sample_rate: Audio sample rate
            p: Probability of applying the augmentation
        """
        super().__init__()
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.sample_rate = sample_rate
        self.p = p

    def __call__(self, data: torch.Tensor):
        """
        Apply pitch shifting to audio tensor.
        
        Args:
            data: Audio tensor of shape (batch, time) or (time,)
        
        Returns:
            Pitch-shifted audio tensor
        """
        if not len(data.shape) == 1:
            return self(data.unsqueeze(0)).squeeze(0)
        
        if torch.rand(1).item() > self.p:
            return data
        
        original_length = data.shape[-1]
        
        semitones = torch.FloatTensor(1).uniform_(self.min_semitones, self.max_semitones).item()
        ratio = 2.0 ** (semitones / 12.0)
        intermediate_length = int(original_length * ratio)
        
        data_reshaped = data.unsqueeze(1)
        pitch_shifted = torch.nn.functional.interpolate(
            data_reshaped,
            size=intermediate_length,
            mode='linear',
            align_corners=False
        )
        restored = torch.nn.functional.interpolate(
            pitch_shifted,
            size=original_length,
            mode='linear',
            align_corners=False
        )
        restored = restored.squeeze(1)
        
        return restored

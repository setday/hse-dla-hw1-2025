from src.transforms.wav_augs.gain import Gain
from src.transforms.wav_augs.add_background_noise import AddBackgroundNoise
from src.transforms.wav_augs.time_stretch import TimeStretch
from src.transforms.wav_augs.pitch_shift import PitchShift
from src.transforms.wav_augs.bit_regressor import BitRegression
from src.transforms.wav_augs.frequency_masking import FrequencyMasking

__all__ = [
    "Gain",
    "AddBackgroundNoise",
    "TimeStretch",
    "PitchShift",
    "BitRegression",
    "FrequencyMasking",
]

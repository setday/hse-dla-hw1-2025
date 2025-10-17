from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            if path.suffix.lower() in [".mp3", ".wav", ".flac", ".m4a"]:
                entry = {}
                entry["path"] = str(path)
                
                try:
                    audio_tensor, sr = torchaudio.load(str(path))
                    audio_len = audio_tensor.shape[1] / sr
                    entry["audio_len"] = audio_len
                except Exception as e:
                    print(f"Warning: Could not load audio file {path}: {e}")
                    continue
                
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)

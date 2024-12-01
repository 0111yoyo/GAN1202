import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, data_dir, target_length=32000):
        """
        Args:
            data_dir (str): Directory containing audio files.
            target_length (int): Fixed length for all audio clips (in samples).
        """
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]
        self.target_length = target_length

        if not self.files:
            raise ValueError(f"No .wav files found in {data_dir}. Ensure preprocessing was successful.")
        print(f"Loaded {len(self.files)} files from {data_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.files[idx], sr=16000)  # Load and resample to 16 kHz
        audio = self._fix_length(audio)
        return torch.tensor(audio, dtype=torch.float32)

    def _fix_length(self, audio):
        """
        Pads or truncates audio to the target length.
        """
        if len(audio) < self.target_length:
            padding = self.target_length - len(audio)
            return np.pad(audio, (0, padding), mode='constant')
        return audio[:self.target_length]
import torch
from models.generator import WaveGANGenerator
import soundfile as sf
import os
import numpy as np
import shutil
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def generate_audio(generator_path, output_dir, num_samples=10, latent_dim=100, output_size=32000, sample_rate=16000, cutoff_freq=3000):
    """
    Generate audio samples using the trained generator.

    Args:
        generator_path (str): Path to the saved generator model.
        output_dir (str): Directory to save generated audio files.
        num_samples (int): Number of audio samples to generate.
        latent_dim (int): Dimension of the latent space.
        output_size (int): Number of samples in each generated audio file.
        sample_rate (int): Sample rate for output audio.
        cutoff_freq (int): Cutoff frequency for low-pass filter.
    """
    # Clear the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load the trained generator
    generator = WaveGANGenerator(latent_dim, output_size=output_size)
    generator.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))
    generator.eval()

    for i in range(num_samples):
        # Generate a random latent vector
        z = torch.randn(1, latent_dim)

        # Generate audio and convert to NumPy array
        with torch.no_grad():
            audio = generator(z).squeeze().cpu().numpy()

        # Normalize audio to [-1, 1] and clip to valid range
        audio = np.clip(audio, -1, 1)

        # Apply low-pass filter to remove high-frequency noise
        filtered_audio = butter_lowpass_filter(audio, cutoff_freq, sample_rate)

        # Save the generated audio file
        output_path = os.path.join(output_dir, f"generated_{i}.wav")
        sf.write(output_path, filtered_audio, samplerate=sample_rate)
        print(f"Generated audio saved as: {output_path}")

if __name__ == "__main__":
    generate_audio("checkpoints/generator_epoch_19.pth", "generated_audio", num_samples=10)
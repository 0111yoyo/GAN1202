
# ML_WaveGAN

This repository contains the implementation of a **WaveGAN** model for generating audio data. It provides tools for preprocessing, training, generating audio samples, and analyzing training performance.

## Contents

### Main Files
- **`train.py`**: The main script for training the WaveGAN model. It includes generator and discriminator optimization, and logs training metrics to `training_log.csv`.
- **`generate.py`** and **`generate_audio.py`**: Scripts for generating audio samples using the trained generator. These scripts include options for applying low-pass filters to enhance audio quality.
- **`plot_losses.py`**: Visualizes training losses from `training_log.csv` and saves the plot as `loss_curve.png`.
- **`preprocess.py`**: Preprocesses raw audio data by converting MP3 files to WAV, normalizing audio, and splitting into fixed-length segments.
- **`preprocess_with_filter.py`**: A more advanced preprocessing script that segments audio and filters it based on the presence of specific features, such as bird sounds, using a pre-trained classifier.

### Supporting Files
- **`loss_curve.png`**: A visualization of the generator and discriminator losses over iterations.
- **`training_log.csv`**: Contains training logs with loss metrics for each epoch and batch.
- **`requirements.txt`**: Lists all Python dependencies required for running the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ml_wavegan.git
   cd ml_wavegan
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `data/` directory exists with appropriate subdirectories:
   - `data/raw/` for unprocessed audio files.
   - `data/processed/` for preprocessed audio files.

## Workflow

### 1. Data Preprocessing
Use the preprocessing scripts to prepare the dataset:
- Basic preprocessing: Run `preprocess.py`.
- Advanced filtering: Run `preprocess_with_filter.py` for feature-specific segmentation.

### 2. Training
Train the model using `train.py`:
```bash
python train.py
```
Training metrics are logged in `training_log.csv`, and checkpoints are saved in the `checkpoints/` directory.

### 3. Loss Visualization
Plot the generator and discriminator losses:
```bash
python plot_losses.py
```
This generates `loss_curve.png`, providing insights into the training dynamics.

### 4. Audio Generation
Generate synthetic audio samples using the trained generator:
```bash
python generate.py
```



## Project Features

- **GAN-based audio generation**: Employs WaveGAN for realistic waveform generation.
- **Flexible preprocessing**: Supports MP3 to WAV conversion, normalization, and feature-based segmentation.
- **Visualization**: Loss curves to monitor training performance.
- **Filtered generation**: Low-pass filtering for noise reduction in generated audio.

## Dependencies
The project requires the following Python libraries:
- `torch`
- `torchaudio`
- `numpy`
- `librosa`
- `matplotlib`
- `pydub`
- `scipy`
- `soundfile`
- `transformers` (used in `preprocess_with_filter.py`)



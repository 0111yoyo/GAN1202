import os
import shutil
import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, ASTForAudioClassification

# Constants
SAMPLE_RATE = 16000
SEGMENT_SECONDS = 1  # Length of each segment in seconds
STRIDE_SECONDS = 0.5  # Overlapping stride in seconds
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
FILTERED_DIR = "data/processed"
MAX_SEGMENTS = 10  # Maximum segments to process from each audio file

def process_audio(audio_path: str):
    """
    Load one .mp3 file into an array.
    Parameters:
        audio_path (str): Path to audio file.

    Returns:
        tuple: (waveform, sample rate)
    """
    waveform, original_sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=SAMPLE_RATE)
    waveform = resampler(waveform)

    return waveform, SAMPLE_RATE

def crop_and_filter_audio(input_dir: str, output_dir: str, segment_seconds=SEGMENT_SECONDS, stride_seconds=STRIDE_SECONDS, max_segments=MAX_SEGMENTS, model_name=MODEL_NAME):
    """
    Crop audio into fixed-length segments and filter segments containing bird sounds.
    Refresh the output directory before processing.
    """
    # Clear the output directory
    if os.path.exists(output_dir):
        print(f"Clearing directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load model and feature extractor
    model = ASTForAudioClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for audio_file in os.listdir(input_dir):
        if audio_file.endswith(".mp3"):
            audio_path = os.path.join(input_dir, audio_file)
            waveform, sample_rate = process_audio(audio_path)

            segment_samples = int(segment_seconds * sample_rate)  # Convert to integer
            stride_samples = int(stride_seconds * sample_rate)    # Convert to integer

            processed_segments = 0  # Track number of saved segments

            for i, start in enumerate(range(0, waveform.shape[1] - segment_samples, stride_samples)):
                if processed_segments >= max_segments:  # Stop if max_segments is reached
                    break

                segment = waveform[:, start:start + segment_samples]
                if segment.shape[1] < segment_samples:
                    segment = F.pad(segment, (0, segment_samples - segment.shape[1]))

                # Classify the segment
                if classify_segment(segment, sample_rate, model, feature_extractor, device):
                    # Save the segment if it's classified as a bird sound
                    filename = f"{os.path.splitext(audio_file)[0]}_seg_{i}.wav"
                    segment_path = os.path.join(output_dir, filename)
                    torchaudio.save(segment_path, segment.cpu(), sample_rate=sample_rate)
                    print(f"Saved bird sound segment: {segment_path}")
                    processed_segments += 1

def classify_segment(segment, sample_rate, model, feature_extractor, device):
    """
    Classify a single audio segment.
    """
    inputs = feature_extractor(
        segment.squeeze().cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        predicted_label = model.config.id2label[predicted_class_id]

    return "bird" in predicted_label.lower()

if __name__ == "__main__":
    RAW_DATA_DIR = "data/raw"
    print("Cropping and filtering audio...")
    crop_and_filter_audio(RAW_DATA_DIR, FILTERED_DIR)
    print("Preprocessing complete. Filtered audio saved in 'data/processed'.")

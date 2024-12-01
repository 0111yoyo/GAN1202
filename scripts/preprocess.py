from pydub import AudioSegment
import os

def preprocess_audio(input_dir, output_dir, segment_length=2000):
    """
    Preprocess audio files:
    - Convert MP3 to WAV
    - Normalize volume
    - Split into segments

    Args:
        input_dir (str): Directory containing MP3 files.
        output_dir (str): Directory to save processed WAV files.
        segment_length (int): Segment length in milliseconds.
    """
    # Ensure directories exist
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Get all MP3 files in the input directory
    files = [file for file in os.listdir(input_dir) if file.endswith(".mp3")]
    if not files:
        print(f"No MP3 files found in '{input_dir}'.")
        return

    processed_count = 0

    for file in files:
        print(f"Processing {file}...")
        try:
            # Load MP3 file
            audio = AudioSegment.from_file(os.path.join(input_dir, file))
            audio = audio.set_frame_rate(16000).set_channels(1)  # Resample and mono
            audio = audio.normalize()  # Normalize volume

            # Split into segments
            for i in range(0, len(audio), segment_length):
                segment = audio[i:i + segment_length]
                output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_seg{i}.wav")
                segment.export(output_file, format="wav")
                processed_count += 1
                print(f"Exported: {output_file}")

        except Exception as e:
            print(f"Failed to process {file}: {e}")

    print(f"Total processed files: {processed_count}")

if __name__ == "__main__":
    preprocess_audio("data/raw", "data/processed")

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load training log
log_file = "training_log.csv"

# Check if the log file exists
if not os.path.exists(log_file):
    raise FileNotFoundError(f"The log file {log_file} does not exist.")

# Read the CSV file
try:
    df = pd.read_csv(log_file)
except Exception as e:
    raise ValueError(f"Error reading the log file: {e}")

# Check if the required columns exist
if "Gen Loss" not in df.columns or "Disc Loss" not in df.columns:
    raise ValueError("The log file must contain 'Gen Loss' and 'Disc Loss' columns.")

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(df["Gen Loss"], label="Generator Loss", alpha=0.7)
plt.plot(df["Disc Loss"], label="Discriminator Loss", alpha=0.7)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()
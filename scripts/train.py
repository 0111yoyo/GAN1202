import time
import os
import csv
import torch
from torch.utils.data import DataLoader
from models.generator import WaveGANGenerator
from models.discriminator import WaveGANDiscriminator
from utils.dataset import AudioDataset

# Hyperparameters
latent_dim = 100
output_size = 32000  # 2 seconds of audio at 16 kHz
epochs = 20
batch_size = 4
learning_rate = 0.0002

# Dataset and DataLoader
dataset = AudioDataset("data/processed", target_length=output_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = WaveGANGenerator(latent_dim, output_size)
discriminator = WaveGANDiscriminator(output_size)

# Optimizers
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = torch.nn.BCELoss()

# Create checkpoints directory if it doesn't exist
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

# Create or open the log file
log_file = "training_log.csv"
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Batch", "Gen Loss", "Disc Loss"])

# Training loop
for epoch in range(epochs):
    start_time = time.time()  # Start timing for this epoch
    for batch_idx, real_audio in enumerate(dataloader):
        batch_size = real_audio.size(0)
        real_audio = real_audio.view(batch_size, -1).float()

        # Train discriminator
        disc_optimizer.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Real audio loss
        real_output = discriminator(real_audio)
        real_loss = criterion(real_output, real_labels)

        # Generate fake audio
        z = torch.randn(batch_size, latent_dim)
        fake_audio = generator(z)
        fake_output = discriminator(fake_audio.detach())
        fake_loss = criterion(fake_output, fake_labels)

        # Total discriminator loss
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        disc_optimizer.step()

        # Train generator
        gen_optimizer.zero_grad()
        fake_output = discriminator(fake_audio)
        gen_loss = criterion(fake_output, real_labels)
        gen_loss.backward()
        gen_optimizer.step()

        # Log losses
        print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Gen Loss: {gen_loss.item()}, Disc Loss: {disc_loss.item()}")

        # Write losses to log file
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, batch_idx, gen_loss.item(), disc_loss.item()])

    # Save model checkpoints
    torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch}.pth")

    print(f"Epoch [{epoch}/{epochs}] completed in {time.time() - start_time:.2f} seconds")
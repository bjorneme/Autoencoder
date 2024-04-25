import torch
import torch.nn as nn

# Encoder class
class Encoder(nn.Module):
    def __init__(self, channels, latent_dimension):
        super(Encoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Input: [batch_size, 1, 28, 28] for MNIST images
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 16, 14, 14]
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 32, 7, 7]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 64, 4, 4]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4),  # Output: [batch_size, 128, 1, 1]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten(),
        )

        # VAE mean and logvar layer
        self.mean = nn.Linear(128, latent_dimension)
        self.log_variance = nn.Linear(128, latent_dimension)

    def forward(self, x):
        # Forward pass Encoder
        x = self.encoder(x)
        return self.mean(x), self.log_variance(x)
    
# Decoder class
class Decoder(nn.Module):
    def __init__(self, channels, latent_dimension):
        super(Decoder, self).__init__()

        # Decoder
        self.linear_decoder = nn.Sequential(
            nn.Linear(latent_dimension, 128), # Output: [batch_size, 128]
            nn.ReLU()
        )

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4),  # Output: [batch_size, 64, 4, 4]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1), # Output: [batch_size, 32, 7, 7]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: [batch_size, 16, 14, 14]
            nn.ReLU(),

            nn.ConvTranspose2d(16, channels, kernel_size=3, stride=2, padding=1, output_padding= 1), # Output: [batch_size, 1, 28, 28]
            nn.Sigmoid()  # Sigmoid to output pixel between 0 and 1

        )

    def forward(self, x):
        # Forward pass Decoder
        x = self.linear_decoder(x)
        x = self.conv_decoder(x.reshape(x.shape[0], 128, 1, 1))
        return x

# VAE class
class VAE(nn.Module):
    def __init__(self, channels=1, latent_dimension=16):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = Encoder(channels, latent_dimension)

        # Decoder
        self.decoder = Decoder(channels, latent_dimension)

    def forward(self, x):
        # Forward pass VAE
        mean, log_variance = self.encoder(x)
        z = self.reparameterize(mean, log_variance)
        x = self.decoder(z)
        return x, mean, log_variance
    
    def reparameterize(self, mean, log_variance):
        # Calculate the std from log variance
        std = torch.exp(0.5 * log_variance)

        # Sample a noise vector from a standard normal distribution
        eps = torch.randn_like(std)

        return mean + eps * std
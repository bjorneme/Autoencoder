import torch.nn as nn

# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, channels=1):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Input size: [batch_size, 1, 28, 28] for MNIST images
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),  # Output size: [batch_size, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output size: [batch_size, 32, 7, 7]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7)  # Output size: [batch_size, 64, 1, 1] - Compressed representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),  # Output size: [batch_size, 32, 7, 7]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: [batch_size, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: [batch_size, 1, 28, 28]
            nn.Sigmoid()  # Using Sigmoid to output pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
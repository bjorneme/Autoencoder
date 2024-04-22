import torch.nn as nn

# Encoder class
class Encoder(nn.Module):
    def __init__(self, channels, latent_dimension):
        super(Encoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Input size: [batch_size, 1, 28, 28] for MNIST images
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),  # Output size: [batch_size, 16, 14, 14]
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output size: [batch_size, 32, 7, 7]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output size: [batch_size, 64, 4, 4]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4),  # Output size: [batch_size, 128, 1, 1]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128, latent_dimension)
        )

    def forward(self, x):
        # Forward pass Encoder
        x = self.encoder(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, channels, latent_dimension):
        super(Decoder, self).__init__()

        # Decoder
        self.linear_decoder = nn.Sequential(
            nn.Linear(latent_dimension, 128), # Output size: [batch_size, 128]
            nn.ReLU()
        )

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4),  # Output size: [batch_size, 64, 4, 4]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1), # Output size: [batch_size, 32, 7, 7]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # Output size: [batch_size, 16, 14, 14]
            nn.ReLU(),

            nn.ConvTranspose2d(16, channels, kernel_size=3, stride=2, padding=1, output_padding= 1), # Output size: [batch_size, 1, 28, 28]
            nn.Sigmoid()  # Using Sigmoid to output pixel values between 0 and 1

        )

    def forward(self, x):
        # Forward pass Decoder
        x = self.linear_decoder(x)
        x = self.conv_decoder(x.reshape(x.shape[0], 128, 1, 1))
        return x

# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, channels=1, latent_dimension=16):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = Encoder(channels, latent_dimension)

        # Decoder
        self.decoder = Decoder(channels, latent_dimension)

    def forward(self, x):
        # Forward pass Autoencoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x
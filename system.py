import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from autoencoder.autoencoder import Autoencoder
from autoencoder.vae import VAE
from stacked_mnist_tf import StackedMNISTData
from verification_net import VerificationNet
from visualization import plot_results


# System class
class System:
    def __init__(self, model_type, data_mode, latent_dimension, model_filepath = None):
        # Initialize latent dimension
        self.latent_dimension = latent_dimension

        # Initialize model
        self.model_type = model_type
        self.data_mode = data_mode
        self.model = self.initialize_model()

        # Initialize data configuration
        self.data_generator = StackedMNISTData(self.data_mode)
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.load_data()

        # Initialize verification network
        self.verification_net = VerificationNet(force_learn=False)
        if not self.verification_net.done_training:
            print("Training the verification network...")
            self.verification_net.train(generator=self.data_generator, epochs=10)

        # Initialize tolerace
        self.tolerance: float = 0.5 if 'COLOR' in self.data_mode.name else 0.8

        # Load a pretrained model
        self.model_filepath = model_filepath
        if self.model_filepath:
            self.load_model(self.model_filepath)
        

    def initialize_model(self):
        # Determine number of channels based on data_mode (COLOR vs. MONO)
        if 'COLOR' in self.data_mode.name:
            channels = 3
        else: channels = 1

        # Create model with specific number of channels
        if self.model_type == 'AE':
            return Autoencoder(channels, self.latent_dimension)
        if self.model_type == 'VAE':
            return VAE(channels, self.latent_dimension)
        else:
            raise ValueError("Unsupported model type")
        
    def load_data(self):
        # Load training and testing data using the specified data mode
        train_images, train_labels = self.data_generator.get_full_data_set(training=True)
        test_images, test_labels = self.data_generator.get_full_data_set(training=False)

        # Rearrange image tensor dimensions: [batch_size, channels, 28, 28]
        train_images = torch.Tensor(train_images).permute(0, 3, 1, 2)
        test_images = torch.Tensor(test_images).permute(0, 3, 1, 2)

        return train_images, train_labels, test_images, test_labels


    def train(self, epochs=10, batch_size=256):
        # Set model to training mode
        self.model.train()

        # Initialize loss function
        if self.model_type == 'AE':
            loss_function = torch.nn.BCELoss()
        elif self.model_type == 'VAE':
            loss_function = self.vae_loss

        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Prepare dataloaders for training and validation sets
        train_loader = DataLoader(TensorDataset(self.train_images, self.train_images), batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(TensorDataset(self.test_images, self.test_images), batch_size=batch_size, shuffle=False)

        # Loop over each model to train the model
        for epoch in range(epochs):
            total_loss = 0

            # Loop over each batch from training loader
            for inputs, targets in train_loader:

                # Reset optimizer
                optimizer.zero_grad()

                if self.model_type == 'AE':
                    # Forward pass
                    outputs = self.model(inputs)

                    # Compute loss
                    loss = loss_function(outputs, targets)

                elif self.model_type == 'VAE':
                    # Forward pass
                    outputs, mean, logvar = self.model(inputs)

                    # Compute loss
                    loss = self.vae_loss(outputs, targets, mean, logvar)

                # Backpropagation
                loss.backward()

                # Update the model parameters
                optimizer.step()

                total_loss += loss.item()
            
            # Validation
            val_loss = self.validation(validation_loader, loss_function)
            
            # Print losses for the current epoch
            print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss/len(validation_loader)}')

        # Save the model
        if self.model_filepath:
            self.save_model()


    def vae_loss(self, outputs, targets, mean, log_variance):
        # Compute the binary cross-entropy loss between the reconstructed images and the original images
        loss = torch.nn.functional.binary_cross_entropy(outputs, targets, reduction='sum')

        # Compute the KL divergence
        kld = -0.5 * torch.sum(1 + log_variance - mean**2 - log_variance.exp())

        return loss + kld

    def validation(self, validation_loader, loss_function):
        # Set model to evaluation mode
        self.model.eval()
        val_loss = 0

        # Disable gradients
        with torch.no_grad():

            # Loop over validation data
            for inputs, targets in validation_loader:
  
                if self.model_type == 'VAE':
                    # Forward pass
                    outputs, mean, logvar = self.model(inputs)

                    # Calculate loss
                    loss = self.vae_loss(outputs, targets, mean, logvar)

                elif self.model_type == 'AE':
                    # Forward pass
                    outputs = self.model(inputs)

                    # Calculate loss
                    loss = loss_function(outputs, targets)

                val_loss += loss.item()

        # Return total validation loss
        return val_loss

    def evaluation(self):
        # Set model to evaluation mode
        self.model.eval()

        # Disable gradient calculations
        with torch.no_grad():

            if self.model_type == 'VAE':
                # Create reconstructions from test images
                reconstructions, _, _ = self.model(self.test_images)
            else:
                # Create reconstructions from test images
                reconstructions = self.model(self.test_images)

            # Evaluate using the VerificationNet
            predictability, accuracy = self.verification_net.check_predictability(reconstructions.permute(0, 2, 3, 1).numpy(), self.test_labels, self.tolerance)
            print(f"Predictability: {predictability:.4f}, Accuracy: {accuracy:.4f}")

        # Plot orginal vs. reconstructions
        plot_results(self.test_images[:10], reconstructions[:10], compare=True)


    def generate_images(self, plot_samples = 10, num_samples=10000):

        # Generate new images from random latent vectors
        self.model.eval()

        # Disable gradients
        with torch.no_grad():

            # Generate latent vector
            z = torch.randn(num_samples, self.latent_dimension)

            # Send the latent vector through the decoder
            generated_images = self.model.decoder(z)

            # Plot generated images
            plot_results(generated_images[:plot_samples], compare=False)

        # Evaluate quality and coverage of generated images
        self.evaluate_images(generated_images.permute(0, 2, 3, 1).numpy())


    def anomaly_detection_ae(self, top_k=10):
        # Set model to evaluation mode
        self.model.eval()

        # Disable gradients
        with torch.no_grad():

            # Prepare the data loader for the test dataset
            test_loader = DataLoader(TensorDataset(self.test_images, self.test_images), batch_size=1, shuffle=False)
            anomaly_scores = []

            # Loop over the test loader and calculate loss
            for images, _ in test_loader:
                outputs = self.model(images)
                loss = torch.nn.functional.mse_loss(outputs, images).item()
                anomaly_scores.append(loss)

            # Find index 
            top_k_indexes  = np.argsort(anomaly_scores)[-top_k:]
            top_images = self.test_images[top_k_indexes]
            reconstructions = self.model(top_images)

            plot_results(top_images, reconstructions, compare=True)


    def evaluate_images(self, generated_images):
        # Evaluate coverage: diversity
        coverage = self.verification_net.check_class_coverage(generated_images, self.tolerance)
        print(f"Coverage: {coverage:.4f}")

        # Evaluate predictability: quality
        predictability, _ = self.verification_net.check_predictability(generated_images, None, self.tolerance)
        print(f"Predictability: {predictability:.4f}")
        
    
    #===========================================================================
    # Save and Load model
    def save_model(self):
        # Saves the trained model
        torch.save(self.model.state_dict(), self.model_filepath)
        print(f"Model saved to: {self.model_filepath}")

    def load_model(self, model_filepath):
        # Loads a pre-trained model
        try:
            self.model.load_state_dict(torch.load(model_filepath))
            self.model.eval()
            print(f"Model loaded from: {model_filepath}")
        except FileNotFoundError:
            print(f"Pre-trained not found: {model_filepath}")
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from autoencoder.autoencoder import Autoencoder
from autoencoder.vae import VAE
from stacked_mnist_tf import StackedMNISTData
from verification_net import VerificationNet


# System class
class System:
    def __init__(self, model_type, data_mode, latent_dimension):
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


    def vae_loss(self, outputs, targets, mean, logvar):
        # Compute the binary cross-entropy loss between the reconstructed images and the original images
        loss = torch.nn.functional.binary_cross_entropy(outputs, targets, reduction='sum')

        # Compute the Kullback-Leibler divergence (KLD) between the learned latent distribution and the prior
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

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

        # Plot result
        self.plot_results(self.test_images[:10], reconstructions[:10])


    def plot_results(self, original, reconstructed, num_images = 10):
        # Rearrange image tensor dimensions: [batch_size, 28, 28, channels]
        original = original.permute(0, 2, 3, 1).numpy()
        reconstructed = reconstructed.permute(0, 2, 3, 1).numpy()

        plt.figure(figsize=(20, 4))
        for i in range(num_images):
            # Display original
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(original[i], cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(reconstructed[i], cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


    def generate_images(self, num_samples=10):

        # Generate new images from random latent vectors
        self.model.eval()

        # Disable gradients
        with torch.no_grad():

            # Generate latent vector
            z = torch.rand(10000, self.latent_dimension)

            # Send the latent vector through the decoder
            generated_images = self.model.decoder(z)

            # Reshape the generated images for plotting and evaluation
            generated_images = generated_images.permute(0, 2, 3, 1).numpy()

            # Display generated images
            plt.figure(figsize=(20, 4))
            for i in range(num_samples):
                ax = plt.subplot(1, num_samples, i + 1)
                plt.imshow(generated_images[i].squeeze(), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()

        # Evaluate quality and coverage of generated images
        self.evaluate_images(generated_images)


# TODO ============================================================
# Implement support for VAE

    def anomaly_detection(self, top_k=10):

        # Set model to evaluation mode
        self.model.eval()

        # Disable gradients
        with torch.no_grad():

            # Prepare the data loader for the test dataset
            test_loader = DataLoader(TensorDataset(self.test_images, self.test_images), batch_size=1, shuffle=False)
            anomaly_scores = []

            # Loop over the test loader and calculate loss
            for data in test_loader:
                images, _ = data

                if self.model_type == 'VAE':
                    outputs, _, _ = self.model(images)
                else:
                    outputs = self.model(images)

                loss = torch.nn.functional.mse_loss(outputs, images, reduction='none').mean([1, 2, 3]).numpy()
                anomaly_scores.extend(loss)

            # Identifying the top_k anomalies
            top_k_indices = np.argsort(anomaly_scores)[-top_k:]
            top_k_anomalies = self.test_images[top_k_indices]
            self.plot_results(top_k_anomalies, self.model(top_k_anomalies))


    def evaluate_images(self, generated_images):
        # Evaluate coverage: diversity
        coverage = self.verification_net.check_class_coverage(generated_images, self.tolerance)
        print(f"Coverage: {coverage:.4f}")

        # Evaluate predictability: quality
        predictability, _ = self.verification_net.check_predictability(generated_images, None, self.tolerance)
        print(f"Predictability: {predictability:.4f}")

        

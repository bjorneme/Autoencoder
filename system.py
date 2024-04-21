import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from autoencoder import Autoencoder
from stacked_mnist_tf import StackedMNISTData

class System:
    def __init__(self, model_type, data_mode):
        # Initialize model
        self.model_type = model_type
        self.data_mode = data_mode
        self.model = self.initialize_model()

        # Initialize data configuration
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.load_data()

    def initialize_model(self):
        # Determine number of channels based on data_mode (COLOR vs. MONO)
        if 'COLOR' in self.data_mode.name:
            channels = 3
        else: channels = 1

        # Create model with specific number of channels
        if self.model_type == 'AE':
            return Autoencoder(channels)
        else:
            raise ValueError("Unsupported model type")
        
    def load_data(self):
        # Load training and testing data using the specified data mode
        data_generator = StackedMNISTData(self.data_mode)
        train_images, train_labels = data_generator.get_full_data_set(training=True)
        test_images, test_labels = data_generator.get_full_data_set(training=False)

        # Rearrange image tensor dimensions: [batch_size, channels, 28, 28]
        train_images = torch.Tensor(train_images).permute(0, 3, 1, 2)
        test_images = torch.Tensor(test_images).permute(0, 3, 1, 2)
        return train_images, train_labels, test_images, test_labels

    def train(self, epochs=10, batch_size=256):
        # Set model to training mode and define loss function and optimizer
        self.model.train()
        criterion = torch.nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        train_loader = DataLoader(TensorDataset(self.train_images, self.train_images), batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for data in train_loader:
                inputs, targets = data
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

    def evaluation(self):
        # Set model to evaluation mode and disable gradient calculations
        self.model.eval()
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(self.test_images, self.test_images), batch_size=10, shuffle=True)
            data_iter = iter(test_loader)
            images, _ = next(data_iter)
            outputs = self.model(images)

            # Plot original and reconstructed images
            self.plot_results(images, outputs)

    def plot_results(self, original, reconstructed, num_images = 8):
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

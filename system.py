import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from autoencoder import Autoencoder
from stacked_mnist_tf import StackedMNISTData
from verification_net import VerificationNet

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
        


    def initialize_model(self):
        # Determine number of channels based on data_mode (COLOR vs. MONO)
        if 'COLOR' in self.data_mode.name:
            channels = 3
        else: channels = 1

        # Create model with specific number of channels
        if self.model_type == 'AE':
            return Autoencoder(channels, self.latent_dimension)
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
        # Set model to training mode and define loss function and optimizer
        self.model.train()
        loss_function = torch.nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        train_loader = DataLoader(TensorDataset(self.train_images, self.train_images), batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(TensorDataset(self.test_images, self.test_images), batch_size=batch_size, shuffle=False)

        # Training Loop
        for epoch in range(epochs):
            total_loss = 0
            for data in train_loader:
                inputs, targets = data
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            val_loss = self.validation(validation_loader, loss_function)
            
            print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss/len(validation_loader)}')
        
    def validation(self, loader, loss_function):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in loader:
                inputs, targets = data
                outputs = self.model(inputs)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
        return val_loss

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
        with torch.no_grad():
            z = torch.rand(num_samples, self.latent_dimension)
            generated_images = self.model.decoder(z)
            generated_images = generated_images.permute(0, 2, 3, 1).numpy()
            plt.figure(figsize=(20, 4))
            for i in range(num_samples):
                ax = plt.subplot(1, num_samples, i + 1)
                plt.imshow(generated_images[i].squeeze(), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()

        self.evaluate_images(generated_images)

    def anomaly_detection(self, top_k=10):
        # Detect anomalies based on reconstruction loss, showing the top k anomalous images
        self.model.eval()
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(self.test_images, self.test_images), batch_size=1, shuffle=False)
            anomaly_scores = []
            for data in test_loader:
                images, _ = data
                outputs = self.model(images)
                loss = torch.nn.functional.mse_loss(outputs, images, reduction='none').mean([1, 2, 3]).numpy()
                anomaly_scores.extend(loss)

            # Identifying the top_k anomalies
            top_k_indices = np.argsort(anomaly_scores)[-top_k:]
            top_k_anomalies = self.test_images[top_k_indices]
            self.plot_results(top_k_anomalies, self.model(top_k_anomalies))

    def evaluate_images(self, generated_images):
        # Evaluate quality and diveristy of the model
        coverage = self.verification_net.check_class_coverage(generated_images)
        predictability, _ = self.verification_net.check_predictability(generated_images)
        print(f"Coverage: {coverage * 100:.2f}%")
        print(f"Predictability: {predictability * 100:.2f}%")

        

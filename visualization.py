from matplotlib import pyplot as plt

# Plotting
def plot_results(original, reconstructed=None, num_images=10, plot_type='comparison'):
    plt.figure(figsize=(20, 4))
    
    # Compare two images
    if plot_type == 'comparison':
        original = original.permute(0, 2, 3, 1).numpy()
        reconstructed = reconstructed.permute(0, 2, 3, 1).numpy()
        for i in range(num_images):
            # Display original
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(original[i].squeeze(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # Display reconstruction
            ax = plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(reconstructed[i].squeeze(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # Only plot the images, no comparison
    elif plot_type == 'single':
        original = original.permute(0, 2, 3, 1).numpy()
        for i in range(num_images):
            ax = plt.subplot(1, num_images, i + 1)
            plt.imshow(original[i].squeeze(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
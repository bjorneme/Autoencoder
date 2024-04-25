from matplotlib import pyplot as plt

# Function used for plotting
def plot_results(original, reconstructed=None, num_images=10, compare=False):

    # Reshape to correct format
    original = original.permute(0, 2, 3, 1).numpy()
    if compare:
        reconstructed = reconstructed.permute(0, 2, 3, 1).numpy()
    
    # Set the number of rows in the plot based on whether comparison is needed
    rows = 2 if compare else 1

    # Initialize the plot
    plt.figure(figsize=(12, 2 * rows))
    
    for i in range(num_images):
        # Plot original images
        ax = plt.subplot(rows, num_images, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')
        ax.axis('off')
        
        if compare:
            # Plot reconstructed images below originals for compariso
            ax = plt.subplot(rows, num_images, num_images + i + 1)
            plt.imshow(reconstructed[i].squeeze(), cmap='gray')
            ax.axis('off')

    # Display the plot    
    plt.show()

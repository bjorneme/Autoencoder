import json
from stacked_mnist_tf import DataMode
from system import System

if __name__ == "__main__":
    # Load configuration from JSON file
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Initialize the System with parameters from the config
    system = System(
        model_type=config['model_type'],
        data_mode=getattr(DataMode, config['data_mode']),
        latent_dimension=config['latent_dimension'],
        model_filepath=config['model_filepath']
    )

    # Train the model
    system.train(epochs=config['epochs'])

    # AE-BASIC and VAE-BASIC
    if config['task_type'] == 'BASIC':
        system.evaluation()

    # AE-GEN and VAE-GEN
    if config['task_type'] == 'GEN':
        system.generate_images()

    # AE-ANOM 
    if config['task_type'] == 'ANOM' and config['model_type'] == 'AE':
        system.anomaly_detection_ae()

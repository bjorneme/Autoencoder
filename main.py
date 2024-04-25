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
        latent_dimension=config['latent_dimension']
    )

    # Train and evaluate the model
    system.train(epochs=config['epochs'])

    # AE-Basic
    if config['task_type'] == 'AE-BASIC':
        system.evaluation()

    # AE-GEN
    if config['task_type'] == 'AE-GEN':
        system.generate_images()

    # AE-ANOM
    if config['task_type'] == 'AE-ANOM' and config['model_type'] == 'AE':
        system.anomaly_detection_ae()

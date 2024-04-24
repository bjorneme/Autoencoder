from stacked_mnist_tf import DataMode
from system import System


if __name__ == "__main__":
    system = System(
        model_type='VAE',
        data_mode=DataMode.COLOR_BINARY_COMPLETE,
        latent_dimension=64
    )
    system.train(epochs=1)
    system.anomaly_detection_vae()
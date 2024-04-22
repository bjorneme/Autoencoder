from stacked_mnist_tf import DataMode
from system import System


if __name__ == "__main__":
    system = System(
        model_type='AE',
        data_mode=DataMode.MONO_BINARY_COMPLETE,
        latent_dimension=16
    )
    system.train(epochs=10)
    system.evaluation()
    system.generate_images()
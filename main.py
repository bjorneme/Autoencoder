from stacked_mnist_tf import DataMode
from system import System


if __name__ == "__main__":
    system = System('AE', DataMode.MONO_BINARY_MISSING)
    system.train(epochs=5)
    system.evaluation()
    system.generate_images()
    system.anomaly_detection()
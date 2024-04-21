from stacked_mnist_tf import DataMode
from system import System


if __name__ == "__main__":
    system = System('AE', DataMode.MONO_BINARY_COMPLETE)
    system.train(epochs=5)
    system.evaluation()
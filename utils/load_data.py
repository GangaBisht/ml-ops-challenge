from tensorflow.keras.datasets import fashion_mnist
from PIL import Image
import numpy as np

def get_test_image():
    """
    Load a test image from Fashion MNIST dataset.
    
    Returns:
    - test_image: PIL Image object.
    """
    (_, _), (x_test, y_test) = fashion_mnist.load_data()
    test_image = Image.fromarray((x_test[0] * 255).astype(np.uint8))
    return test_image

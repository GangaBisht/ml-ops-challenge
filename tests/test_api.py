import requests
import numpy as np
from PIL import Image
import io
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST test images
(_, _), (x_test, y_test) = fashion_mnist.load_data()

# Loop through the first 5 test images
for i in range(5):
    test_image = Image.fromarray((x_test[i] * 255).astype(np.uint8))

    # Convert the image to bytes
    image_byte_array = io.BytesIO()
    test_image.save(image_byte_array, format='PNG')
    image_byte_array = image_byte_array.getvalue()

    # Make a POST request to the FastAPI endpoint
    response = requests.post("http://127.0.0.1:8000/predict/", files={"file": ("test_image.png", image_byte_array)})

    # Print the response
    print(f"Test Image {i+1}: {response.json()}")

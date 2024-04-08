Fashion MNIST Classifier
This project aims to classify fashion items from the Fashion MNIST dataset using a pre-trained MobileNetV2 model. It provides a FastAPI-based RESTful API for online inference and is containerized using Docker for easy deployment and scalability.

Project Structure
ml-ops-challenge/
│
├── api/
│   ├── __init__.py
│   └── predict.py
│
├── models/
│   └── mobilenetv2.py
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── load_data.py
│
├── tests/
│   └── test_api.py
│
├── main.py
└── requirements.txt
Features
API Development: A FastAPI-based RESTful API with a single endpoint for online inference.
Containerization: The application is containerized using Docker for easy deployment and scalability.
Pre-trained Model: Utilizes a pre-trained MobileNetV2 model for inference.
Data Preprocessing: Includes utility functions for image preprocessing and data loading.
Installation and Setup
Prerequisites
Docker
Python 3.x
Installation Steps
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/fashion_mnist_classifier.git
Navigate to the project directory:

bash
Copy code
cd fashion_mnist_classifier
Build the Docker image:

bash
Copy code
docker build -t ml-ops-challenge .
Run the Docker container:

bash
Copy code
docker run -d -p 8000:80 ml-ops-challenge
Usage
API Endpoint
After running the Docker container, you can access the API at:

bash
Copy code
http://localhost:8000/predict/
Example API Request
You can use the following Python script to send an image to the API for inference:

python
Copy code
import requests
import numpy as np
from PIL import Image
import io

# Load a test image from Fashion MNIST dataset
(_, _), (x_test, y_test) = fashion_mnist.load_data()
test_image = Image.fromarray((x_test[0] * 255).astype(np.uint8))

# Convert the image to bytes
image_byte_array = io.BytesIO()
test_image.save(image_byte_array, format='PNG')
image_byte_array = image_byte_array.getvalue()

# Make a POST request to the FastAPI endpoint
response = requests.post("http://localhost:8000/predict/", files={"file": ("test_image.png", image_byte_array)})

# Print the response
print(response.json())
Testing
You can run the automated tests using:

bash
Copy code
pytest tests/
Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License
This project is licensed under the MIT License. See the LICENSE file for details.

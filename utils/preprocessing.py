from PIL import Image
import numpy as np

def preprocess_image(image):
    """
    Preprocess the input image for prediction.
    
    Parameters:
    - image: PIL Image object.
    
    Returns:
    - image_array: Preprocessed image array.
    """
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return image_array

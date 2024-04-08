from fastapi import APIRouter, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from utils.preprocessing import preprocess_image
from utils.load_data import get_test_image
from models.mobilenetv2 import load_model
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import io

router = APIRouter()

model = load_model()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@router.post("/predict/")
async def predict_endpoint(file: bytes = File(...)):
    """
    Predict the class of the uploaded image using the trained model.
    
    Parameters:
    - file: The uploaded image file (PNG, JPG).
    
    Returns:
    - class_id: The predicted class ID.
    - class_name: The predicted class name.
    """
    try:
        #test_image = get_test_image()
        # image = Image.open(io.BytesIO(file)).convert("L") 
        # image_array = preprocess_image(test_image)
        # image_array = np.expand_dims(image_array, axis=0)
        
        # prediction = model.predict(image_array)
        # predicted_class = np.argmax(prediction, axis=1)[0]
        
        # return {"class_id": int(predicted_class), "class_name": class_names[predicted_class]}

        image = Image.open(io.BytesIO(file)).convert("RGB")
        
        # Preprocess the image
        image_array = preprocess_image(image)
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        return {"class_id": int(predicted_class), "class_name": class_names[predicted_class]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

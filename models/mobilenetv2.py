import tensorflow as tf

def load_model():
    """
    Load the pre-trained MobileNetV2 model for Fashion MNIST classification.
    
    Returns:
    - model: The pre-trained MobileNetV2 model.
    """
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for Fashion MNIST
    ])
    #model.load_weights("../models/mobilenetv2_fashion_mnist.h5")
    
    return model

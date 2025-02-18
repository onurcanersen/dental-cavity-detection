import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load the saved model
model = tf.keras.models.load_model("../models/cavity_model.h5")

# Define class labels
class_labels = {0: "cavity", 1: "healthy"}  # Update based on your dataset order

def predict_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    print(f"Predicted class: {class_labels[predicted_class]}")
    return class_labels[predicted_class]

# Run prediction from command line
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
    else:
        img_path = sys.argv[1]
        predict_image(img_path)

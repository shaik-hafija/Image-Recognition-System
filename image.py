import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load MobileNet model pre-trained on ImageNet dataset
model = MobileNet(weights='imagenet')

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to predict the content of the image
def predict_image(img_path):
    preprocessed_img = preprocess_image(img_path)
    preds = model.predict(preprocessed_img)
    predictions = decode_predictions(preds, top=3)[0]  # Get top 3 predictions
    return [(label, round(score * 100, 2)) for (_, label, score) in predictions]

# Path to your image
img_path = 'path/to/your/image.jpg'

# Make predictions
predictions = predict_image(img_path)

# Print predictions
print("Predictions:")
for label, score in predictions:
    print(f"{label}: {score}%")

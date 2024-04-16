# from flask import Flask,render_template,request
# app = Flask(__name__)

# @app.route('/')
# def man():
#     return render_template('home.html')


# @app.route('/predict',methods=['POST'])
# def home():
#     img=request.form['image']
#     print(img)
#     pred=model.predict(img)
#     return render_template('home.html')

# if __name__=="__main__":
#     # app.run(debug=True)
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
import os


# Load MobileNet model pre-trained on ImageNet dataset
model = MobileNet(weights='imagenet')

# Initialize Flask app
app = Flask(__name__)

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

# Define home route
@app.route('/')
def man():
    return render_template('home.html')

# Define predict route
@app.route('/predict', methods=['POST'])
def home():
    # Get image file from form data
    img_file = request.files['image']
    
    # Save the image to a temporary file
    img_path = 'temp.jpg'
    img_file.save(img_path)
    
    # Make predictions
    predictions = predict_image(img_path)
    
    # Remove the temporary image file
    os.remove(img_path)
    
    # Render predictions
    return render_template('home.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)

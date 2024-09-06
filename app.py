import streamlit as st
import os
os.environ['CV2_CUDNN_STREAM'] = '1'
import cv2
import pandas as pd
import numpy as np
import pickle
import tempfile
from PIL import ImageFont, ImageDraw, Image
from efficientnet.tfkeras import EfficientNetB4
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img


# opening the image
# image = open('banner_image.jpeg', 'rb').read()
# st.image(image, use_column_width=True)

st.title("Croissant")
st.markdown("Predict skin tone")

# Add a separator between the header and the main content
st.markdown("---")



# Load the trained model
model = load_model('model.h5')



st.header("Upload your image")

# Get user input for image upload
uploaded_file = st.file_uploader('Upload an image of your croissant', type=['jpg', 'jpeg', 'png'])

# Process the uploaded image if it exists
if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load new images for prediction
    new_image = image
    new_image_array = np.array(new_image.resize((384, 384)))  # Resize the image to match the model's input shape
    new_image_array = np.expand_dims(new_image_array, axis=0)  # Add batch dimension
    new_image_array = new_image_array / 255.0  # Normalize the pixel values (same as during training)

    # Make predictions
    predictions = model.predict(new_image_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Indicate class names
    class_names = ['Cool', 'Neutral', 'Warm']

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]

    st.write(f"Your croissant is : {predicted_class_name}")



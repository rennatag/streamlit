import streamlit as st
import gdown
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageColor
import os
import tensorflow as tf

# Function to download the file from Google Drive
def download_file_from_drive(url, output):
    gdown.download(url, output, quiet=False)

# Preprocess the uploaded image
def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Make predictions using the loaded model
def predict_image(model, image):
    preds = model.predict(image)
    return preds


# Displays a solid color image based on the provided color code.
def display_color_image(color_code, image_size=(200, 200)):
    try:
        # Convert color code to RGB tuple
        rgb_color = ImageColor.getrgb(color_code)

        # Create a new image with the specified size and color
        img = Image.new("RGB", image_size, rgb_color)

        # Display the image
        st.image(img, caption=f"Color: {color_code}", use_column_width=True)

    except ValueError:
        print(f"Invalid color code: {color_code}")

# Google Drive file link (shared link)
file_url = 'https://drive.google.com/uc?id=16oeHRDRpTGocvN39fz36O67OGURcYS04'
output_file = 'model.keras'

# Streamlit app starts here
st.title("Skintone Predictor")

# Button to trigger model download
#if st.button('Download and Load Model'):
with st.spinner('Model loading. Please wait ...'):
    download_file_from_drive(file_url, output_file)
    st.success('Model downloaded successfully!')

# Check if the file exists
if os.path.exists(output_file):
    # Check if the file is indeed an .h5 file by size (usually > few MBs)
    file_size = os.path.getsize(output_file)
    #st.write(f"Downloaded file size: {file_size / (1024 * 1024)} MB")
    
    if file_size > 0:  # Ensuring the file is not empty
        try:
            with st.spinner('Loading model...'):
                model = load_model(output_file)
                st.success('Model loaded successfully!')
                #st.write(model.summary())
        except OSError as e:
            st.error(f"Error loading model: {e}")
    else:
        st.error('Downloaded file is empty or corrupted.')
else:
    st.error('Error: Model file not found.')

# Image uploader
uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"])

# Once an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image for prediction
    processed_image = preprocess_image(image, target_size=(384, 384))  # Adjust target size to your model input

    # Ensure the model is loaded
    if os.path.exists(output_file):
        try:
            # Load the model
            model = load_model(output_file)

            # Make prediction
            with st.spinner('Making prediction...'):
                predictions = predict_image(model, processed_image)
                st.success('Prediction completed!')

                # Indicate class names
                class_names = ['Cool', 'Neutral', 'Warm']

                # Display prediction
                st.write(f"Predicted skin tone: {class_names[np.argmax(predictions)]}")  # Example prediction format

                # Example usage
                color_code = "#00FF00"  # Green
                display_color_image(color_code)
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.warning("Please download and load the model before making a prediction.")

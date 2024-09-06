import streamlit as st
import gdown
from keras.models import load_model
import os

# Function to download the file from Google Drive
def download_file_from_drive(url, output):
    gdown.download(url, output, quiet=False)

# Google Drive file link (shared link)
file_url = 'https://drive.google.com/uc?id=16oeHRDRpTGocvN39fz36O67OGURcYS04'
output_file = 'model.h5'

# Streamlit app starts here
st.title("Load Keras Model from Google Drive")

# Button to trigger model download
if st.button('Download and Load Model'):
    with st.spinner('Downloading model...'):
        download_file_from_drive(file_url, output_file)
        st.success('Model downloaded successfully!')

    # Check if the file exists
    if os.path.exists(output_file):
        with st.spinner('Loading model...'):
            model = load_model(output_file)
            st.success('Model loaded successfully!')
            st.write(model.summary())
    else:
        st.error('Error: Model file not found.')

st.write("Press the button to download and load the Keras model.")

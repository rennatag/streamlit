import cv2
import tempfile
import streamlit as st
import numpy as np
import csv
from PIL import ImageFont, ImageDraw, Image
import pickle
import pandas as pd
import threading
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CV2_CUDNN_STREAM'] = '1'
from efficientnet.tfkeras import EfficientNetB4
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img

# opening the image
#image = open('banner_image.jpeg', 'rb').read()
#st.image(image, use_column_width=True)

# Load the trained model
model = load_model('model.h5')

    
# Set Class Names
class_labels = ['Cool', 'Neutral', 'Warm']

# Set font
font = ImageFont.truetype(font = "C:/Windows/Fonts/Arial.ttf", size = 30)

st.title("Skin Tone Classifier")
st.markdown("App for classifying skin tone.")

# Add a separator between the header and the main content
st.markdown("---")

# Upload tabs
selected_tab = st.radio("Choose from below options:", ["Upload your Video", "Upload your Image", "Live Video"])

if selected_tab == "Upload your Video":
    st.header("Upload your Video")
    
    
    stframe = st.empty()

    #file uploader
    video_file_buffer = st.file_uploader("Upload an image/video", type=[ "jpeg","jpg","png","mp4", "mov",'avi','asf','m4v'])

    #temporary file name 
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if video_file_buffer:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

        #values 
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc('V','P','0','9')
        out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))


        # alarm properties for alert
        alarm_sound = pyttsx3.init()
        voices = alarm_sound.getProperty('voices')
        alarm_sound.setProperty('voice', voices[0].id)
        alarm_sound.setProperty('rate', 150)

        while vid.isOpened():

            ret, frame = vid.read()
            if ret == False:
                break

            #recoloring it back to BGR b/c it will rerender back to opencv
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image.flags.writeable = True

            # Resize the frame to match the model's input shape (384x384)
            resized_image = cv2.resize(image, (384, 384))

            # Convert the resized image to an array of floating-point numbers
            image_array = np.array(resized_image.astype(np.float32)) / 255.0
            image_array = image_array.reshape((1, 384, 384, 3))

            try:
                # Make prediction
                predictions = model.predict(image_array)
                predicted_class_index = np.argmax(predictions)
                behaviour = class_labels[predicted_class_index]

                #st.write(behaviour)

                # setting image writeable back to true to be able process it
                image.flags.writeable = True
                pil_im = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_im)

                # Print the predicted class on video frame
                draw.text((20,0), behaviour, font = font)
                image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)



                # Play alarm based on 'class'
                if behaviour == 'Cool':
                    alarm_sound.say("Skin Tone is Cool")
                    alarm_sound.runAndWait()
                    if alarm_sound._inLoop:
                        alarm_sound.endLoop()

            except:
                pass                





            # To display the annotated live video feed
            stframe.image(image, use_column_width=True)

        vid.release()
        out.release()
        cv2.destroyAllWindows()


elif selected_tab == "Upload your Image":
    st.header("Upload your Image")
   
    # Load the trained model
    #model = load_model('efficientnet2.h5')

    # Get user input for image upload
    uploaded_file = st.file_uploader('Upload an image of your skin to predict the skin tone', type=['jpg', 'jpeg', 'png'])

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

        st.write(f"Predicted Skin Tone: {predicted_class_name}")


elif selected_tab == "Live Video Analysis":
    # Launch webcam
    stframe = st.empty()
    vid = cv2.VideoCapture(0)

    #values 
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))

    # alarm properties for alert
    alarm_sound = pyttsx3.init()
    voices = alarm_sound.getProperty('voices')
    alarm_sound.setProperty('voice', voices[0].id)
    alarm_sound.setProperty('rate', 150)

    while vid.isOpened():

        ret, frame = vid.read()
        if ret == False:
            break

        #recoloring it back to BGR b/c it will rerender back to opencv
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image.flags.writeable = True

        # Resize the frame to match the model's input shape (384x384)
        resized_image = cv2.resize(image, (384, 384))

        # Convert the resized image to an array of floating-point numbers
        image_array = np.array(resized_image.astype(np.float32)) / 255.0
        image_array = image_array.reshape((1, 384, 384, 3))

        try:
            # Make prediction
            predictions = model.predict(image_array)
            predicted_class_index = np.argmax(predictions)
            behaviour = class_labels[predicted_class_index]

            #st.write(behaviour)

            # setting image writeable back to true to be able process it
            image.flags.writeable = True
            pil_im = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_im)

            # Print the predicted class on video frame
            draw.text((20,0), behaviour, font = font)
            image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)



            # Play alarm based on 'class'
            if behaviour == 'Cool':
                alarm_sound.say("Skin Tone is Cool")
                alarm_sound.runAndWait()
                if alarm_sound._inLoop:
                    alarm_sound.endLoop()

        except:
            pass                


        # To display the annotated live video feed
        stframe.image(image, use_column_width=True)

    vid.release()
    out.release()
    cv2.destroyAllWindows()
       
else:
    st.write("Work in Progress")

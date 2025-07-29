import streamlit.components.v1 as components
from secrets import choice
import streamlit as st

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keras.models import load_model
from scipy.spatial import distance
from imageio import imread
from skimage.transform import resize
import sys
sys.path.append('code/')
import inception_resnet_v1 as Inception

mediapipe_path = 'model/mediapipe/blaze_face_short_range.tflite'
model_path = 'model/facenet_2/model/skenario_2/facenet_keras.keras'
model = load_model(model_path, compile=False, custom_objects={'Custom>scaling': Inception.scaling, 'l2_norm': Inception.l2_norm})

FRAME_WINDOW = st.image([])
menu = ["Login", "Register", "Ujian"]

choice = st.sidebar.selectbox('Menu', menu)

def load_and_align_images(image, margin):
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Create a face detector instance with the image mode:
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_path),
        running_mode=VisionRunningMode.IMAGE)
    with FaceDetector.create_from_options(options) as detector:
      # The detector is initialized. Use it here.
      # ...
        img = imread(image)
        mp_image = mp.Image.create_from_file(image)
        faces = detector.detect(mp_image)
        if len(faces.detections) > 0 :
            bbox = faces.detections[0].bounding_box
            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height
            cropped = img[y-margin//2:y+h+margin//2,
                      x-margin//2:x+w+margin//2, :]
            aligned_images = resize(cropped, (160, 160), mode='reflect')
            
            return aligned_images

def get_embedding(face):
    sample = np.expand_dims(face, axis=0)
    return model.predict(sample)

def calc_dist(embedding1, embedding2):
    return np.linalg.norm(embedding1-embedding2)

# Initialize a flag in session state
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

if st.session_state.form_submitted:
    st.write("You logged in")
    if st.button("Log Out"):
         st.session_state.form_submitted = False
         st.success("Logged out!")
         st.rerun()
         
else:
    st.title("Register")
    with st.form("my_data_entry_form"):
        data_input = st.text_input("Input Name:")
        submitted = st.form_submit_button("Register")

        if submitted:
            # Process data here
            st.session_state.form_submitted = True
            st.success("Register success!")
            st.rerun()


    
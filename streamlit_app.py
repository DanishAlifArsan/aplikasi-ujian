import streamlit.components.v1 as components
from secrets import choice
import streamlit as st

import os
import cv2
import numpy as np
from PIL import Image
# import sys
# sys.path.append('code')
# import controller

# Initialize a flag in session state
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
    
if 'input_name' not in st.session_state:
    st.session_state.input_name = ""

if st.session_state.form_submitted:
    if st.button("Log Out"):
        st.session_state.input_name = ""
        st.session_state.form_submitted = False
        st.success("Logged out!")
        st.rerun()
    
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Camera not available")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    break

            #     face_image, x, y, w, h = controller.load_and_align_videos(ret, frame, cap)
            #     if face_image is not None:
            #         embs = controller.get_embedding(face_image)
            #         identity = ""

            #         list_embs = {}
            #         data = controller.get_data()
                    
            #         for name, db_embs in data.items():
            #             dist = controller.calc_dist(embs, db_embs)
            #             list_embs[name] = dist

            #             name = min(list_embs, key=list_embs.get)
            #             identity = st.session_state.input_name

            #         if name == st.session_state.input_name:
            #             true_data += 1
            #         else:
            #             false_data += 1

            #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #         cv2.putText(frame, identity, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #         cv2.putText(frame, f"True :{true_data}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #         cv2.putText(frame, f"False :{false_data}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #         cv2.putText(frame, f"{list_embs[st.session_state.input_name]:.4f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
                FRAME_WINDOW.image(frame)
            cap.release()
    
else:
    st.subheader("Register")
    with st.form("my_data_entry_form"):

        def load_image(image_file):
            img = Image.open(image_file)
            return img
    
        input_name = st.text_input("Name")
        img_file_buffer = st.camera_input("Take a picture")            
        submitted = st.form_submit_button("Register")

        if submitted:
            if len(input_name) > 0 and img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                path = os.path.join("database",f"{input_name}.jpg")
                success = cv2.imwrite(path, cv2_img)

                if success:
                    st.success(f"Image successfully saved image")
                else:
                    st.error("Error: Could not save the image.")
             
                st.session_state.form_submitted = True
                st.session_state.input_name = input_name
                st.success("Register success!")
                st.rerun()
            else:
                st.error("Please fill in required data!")


    
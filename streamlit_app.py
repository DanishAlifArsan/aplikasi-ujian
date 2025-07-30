import streamlit.components.v1 as components
from secrets import choice
import streamlit as st

import os
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append('code/')
import controller

FRAME_WINDOW = st.image([])

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
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_image = controller.load_and_align_videos(ret, frame, cap)
        identity = ""

        list_embs = {}
        data = controller.get_data()
        for name, db_embs in data.items():
            dist = controller.calc_dist(embs, db_embs)
            list_embs[name] = dist

        name = min(list_embs, key=list_embs.get)
        identity = name

        if name == st.session_state.input_name:
            true_data += 1
        else:
            false_data += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, identity, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"True :{true_data}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"False :{false_data}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"{list_embs[st.session_state.input_name]:.4f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        FRAME_WINDOW.image(frame)
        cv2.waitKey(1)
    

else:
    st.subheader("Register")
    with st.form("my_data_entry_form"):

        def load_image(image_file):
            img = Image.open(image_file)
            return img
    
        input_name = st.text_input("Name")
        image_file = st.file_uploader("Upload Photo",type=['png','jpeg','jpg'])
        submitted = st.form_submit_button("Register")

        if submitted:
            if input_name is not None and image_file is not None:
                file_details = {"FileName":input_name,"FileType":image_file.type}
                st.write(file_details)
                img = load_image(image_file)
                with open(os.path.join("database",image_file.name),"wb") as f: 
                    f.write(image_file.getbuffer())         
                    st.success("Saved File")
        
                st.session_state.form_submitted = True
                st.session_state.input_name = input_name
                st.success("Register success!")
                st.rerun()
            else:
                st.error("Please fill in required data!")


    
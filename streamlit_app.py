import streamlit.components.v1 as components
from secrets import choice
import streamlit as st

import os
import numpy as np
from PIL import Image

# Initialize a flag in session state
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
    
if 'input_name' not in st.session_state:
    st.session_state.input_name = ""

if st.session_state.form_submitted:
    st.write("You logged in")
    st.write(f"Hello {st.session_state.input_name}")
    if st.button("Log Out"):
        st.session_state.input_name = ""
        st.session_state.form_submitted = False
        st.success("Logged out!")
        st.rerun()

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


    
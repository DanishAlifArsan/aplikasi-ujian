import streamlit.components.v1 as components
from secrets import choice
import streamlit as st

import cv2

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
        input_name = st.text_input("Input Name:")
        enable = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable)

        if picture:
            st.image(picture)
        
        submitted = st.form_submit_button("Register")

        if submitted:
            if input_name is not None and picture is not None:
                # Process data here
                bytes_data = picture.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                output_filename = f"{input_name}.png"
                full_output_path = os.path.join('database', output_filename)
                cv2.imwrite(full_output_path, cv2_img)
        
                st.session_state.form_submitted = True
                st.session_state.input_name = input_name
                st.success("Register success!")
                st.rerun()
            else:
                st.error("Please fill in required data!")


    
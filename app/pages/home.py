import streamlit as st

def take_pictures():
    """Take a picture with the webcam.
    Args:
        None
    Returns:
        None
    """
    
    img_file_buffer = st.camera_input("Tire uma foto")

    if img_file_buffer:
        st.image(img_file_buffer)
    pass
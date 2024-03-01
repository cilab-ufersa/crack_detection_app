import streamlit as st
from utils import *

st.set_page_config(
    page_title="Surface Crack Detection",
    page_icon="🧱",
    layout="wide", 
)

st.header("Surface Crack Detection AI App")


left_column, right_column = st.columns(2)

with left_column:
    st.header("Source Image")
    source = st.radio("Get Image from", ["Camera", "Upload"])
    
    if source == "Camera":
        source_img = st.camera_input("Take a picture from your camera")
    elif source == "Upload":
        source_img = st.file_uploader("Upload an image here")


input_directory = source_img
pred = segmentation(input_directory)
        

with right_column:
    st.header("Result Image")
    
    if source_img is not None:
        st.image(pred, use_column_width=True) # showing the segmented image
        st.table({"Prediction": ["Crack", "No Crack"], "Confidence": ["0.8", "0.2"]}) # examples only
    else:
        st.info("Please select an image from the left column")
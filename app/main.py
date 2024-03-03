import streamlit as st
import os
from utils import *
import numpy as np


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


temp_dir = "app/temp"  # input images are saved in this dir
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

input_directory = source_img

img_path = os.path.join(temp_dir, input_directory.name)
# saving the images
with open(img_path, "wb") as f:
    f.write(input_directory.getvalue())

pred = segmentation(img_path)


with right_column:
    st.header("Result Image")

    if source_img is not None:
        st.image(
            (pred[0] * 255).astype(np.uint8), use_column_width=True
        )  # showing the segmented image
        st.table(
            {"Prediction": ["Crack", "No Crack"], "Confidence": ["0.8", "0.2"]}
        )  # examples only
    else:
        st.info("Please select an image from the left column")

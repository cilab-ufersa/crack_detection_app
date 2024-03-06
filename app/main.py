import streamlit as st
import os
from utils import *
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Surface Crack Detection",
    page_icon="🧱",
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


with right_column:
    st.header("Result Image")

    if source_img is not None:
        img_path = os.path.join(temp_dir, input_directory.name)

        # saving the input images
        with open(img_path, "wb") as f:
            f.write(input_directory.getvalue())

        pred = segmentation(img_path)
        segmentaded_array = (pred[0] * 255).astype(np.uint8)
        segmentaded_image = Image.fromarray(np.squeeze(segmentaded_array, axis=2))
        seg_path = os.path.join(temp_dir, f"segmented_{input_directory.name}")

        segmentaded_image.save(seg_path)

        st.image(
            segmentaded_image, use_column_width=True
        )  # showing the segmented image
        
        negative_result, positive_result = classification(img_path)
        
        st.header("Classification Result")
        
        data = {
            "Class": ["Not Containing Crack", "Containing Crack"],
            "Probability": [f"{round(negative_result, 2)}%", f"{round(positive_result, 2)}%"],
        }

        df = pd.DataFrame(data)
        
        st.write(df.reset_index(drop=True).to_html(index=False), unsafe_allow_html=True)
    
    else:
        st.info("Please select an image from the left column")

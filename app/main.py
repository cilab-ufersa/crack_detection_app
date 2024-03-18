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

st.write(
    """
    The model used in this application is a combination of two models: a segmentation model (U-Net) and a classification model (MobileNetV1). 
    With that, the model can detect and classify the presence of cracks in the surface of the material. It achieved an accuracy of 97%, 
    with an F1-score of 0.97 for both classes.
    """
)

st.write("""
         These are the metrics of the model used in this application:
        | Class | Precision | Recall | F1-Score | Support |
        |-------|-----------|--------|----------|---------|
        | Not Containing Crack | 0.95 | 1.00 | 0.97 | 576 |
        | Containing Crack | 1.00 | 0.95 | 0.97 | 624 |
        | Accuracy | | | 0.97 | 1200 |
        | Macro Avg | 0.97 | 0.98 | 0.97 | 1200 |
        | Weighted Avg | 0.98 | 0.97 | 0.97 | 1200 |
         """)


left_column, right_column = st.columns(2)

with left_column:
    st.header("Source Image")
    source = st.radio("Get Image from", ["Camera", "Upload"], horizontal=True)

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

        mask, binary = segmentation(img_path)
        
        mask_path = os.path.join(temp_dir, f"segmented_mask_{input_directory.name}")
        binary_path = os.path.join(temp_dir, f"segmented_binary_{input_directory.name}")

        mask.save(mask_path)
        binary.save(binary_path)
    
        source = st.radio("Result Image as", ["Overlay", "Binary"], horizontal=True)
        
        if source == "Overlay":
            st.image(mask_path, caption="Segmented Image", use_column_width=True)
        else:
            st.image(binary_path, caption="Segmented Image", use_column_width=True)
        
        negative_result, positive_result = classification(img_path)
        
        st.header("Classification Result")
        
        # Displaying the classification result 
        data = {
            "Class": ["Not Containing Crack", "Containing Crack"],
            "Probability": [f"{round(negative_result, 2)}%", f"{round(positive_result, 2)}%"],
        }

        df = pd.DataFrame(data)
              
        st.write(df.reset_index(drop=True).to_html(index=False), unsafe_allow_html=True)
        
        html = create_download_link(save_pdf(image = img_path, overlay = mask_path, binary = binary_path, negative= negative_result, positive=positive_result), "Report")

        st.markdown(html, unsafe_allow_html=True)
        
        
    
    else:
        st.info("Please select an image from the left column")

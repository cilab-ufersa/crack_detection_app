import streamlit as st
import os
from utils import *
import numpy as np
from PIL import Image
import pandas as pd

accuracy = 97
precision = 98
recall = 98


st.set_page_config(
    page_title="Surface Crack Detection",
    page_icon="🧱",
    layout="wide"
)

st.header("Surface Crack Detection AI App 🧱")

first_col, second_col = st.columns([3, 1])
with first_col:
    st.write(
        """
        Crack detection is an important task in the field of civil engineering. Surface cracks can be a sign of structural damage and can lead to catastrophic failure. 
        This app uses a deep learning model to detect cracks in images of concrete surfaces. You can upload an image or take a picture from your camera to see the model in action.
        It will segment the image to highlight the cracks and classify the image as containing a crack or not. You can also download a report of the results.
        """
    )
    
    st.header("Model Performance")
        
    col_1, col_2, col_3 = st.columns(3)
    
    col_1.metric("Accuracy", value=f"{accuracy}"+"%", help="Accuracy of the model: indicates how precise a model's prediction is")
    col_2.metric("Precision", value=f"{precision}"+"%", help="Precision of the model: indicates the model's ability to correctly predict the cases where there was a crack")
    col_3.metric("Recall", value=f"{recall}"+"%", help="Recall of the model: indicates how many of the crack cases the model was able to predict correctly in the dataset")
    
    
    

with second_col:
    # st.image("app/temp/a_0_10.png", caption="Surface Crack", use_column_width=True)
    st.image("app/images/crack.gif", caption="Surface Crack", use_column_width=True)
    
left_column, right_column = st.columns(2)

temp_dir = "app/temp"  # input images are saved in this dir
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)


with left_column:
    st.header("Source Image")
    source = st.radio("Get Image from", ["Camera", "Upload"], horizontal=True)

    if source == "Camera":
        source_img = st.camera_input("Take a picture from your camera")
    elif source == "Upload":
        source_img = st.file_uploader("Upload an image here")
    
    input_directory = source_img
    
    
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
        
        
        negative_result, positive_result = classification(img_path)
        
        
        st.header("Classification Result")
         
        data = {
            "Class": ["Not Containing Crack", "Containing Crack"],
            "Probability": [f"{round(negative_result, 2)}%", f"{round(positive_result, 2)}%"],
        }

        df = pd.DataFrame(data)
              
        st.write(df.reset_index(drop=True).to_html(index=False), unsafe_allow_html=True)
        
        description = st.text_input("Description (Optional)", help="Enter a description of the image, this will be included in the report. Press enter to submit.", placeholder="Example: Crack detected near the window")
        
        if description == "":
            description = "No description provided"
        
        st.header("Downloads")
        characterization_class = characterization(binary_path)
        pdf = save_pdf(image = img_path, overlay = mask_path, binary = binary_path, negative= negative_result, positive=positive_result, user_description=description)

        st.download_button(
            label="Download Result as PDF",
            data=pdf,
            file_name="report.pdf",
            mime="application/pdf",
        )
        
        download_imgs = download_images(image=img_path, overlay=mask_path, binary=binary_path)
        
        with open(download_imgs, "rb") as f:
            st.download_button("Download Images as ZIP", 
                               f, 
                               file_name="images.zip", 
                               mime="application/zip"
                               )
        


with right_column:
    st.header("Result Image")

    if source_img is not None:
        img_path = os.path.join(temp_dir, input_directory.name)

        # saving the input images
        with open(img_path, "wb") as f:
            f.write(input_directory.getvalue())

        source = st.radio("Result Image as", ["Overlay", "Binary"], horizontal=True)
        
        if source == "Overlay":
            st.image(mask_path, caption="Segmented Image", width=400)
        else:
            st.image(binary_path, caption="Segmented Image", width=400)
            
        if negative_result > positive_result:
            st.success("The image does not contain a crack.")
        else:
            st.warning(f"The image contains a crack and it is classified as: {characterization_class}")   
    else:
        st.info("Please select an image from the left column")


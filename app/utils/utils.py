import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from keras.models import load_model, Model
from utils.loss_metrics import (
    Weighted_Cross_Entropy,
    Precision_dil,
    F1_score,
    F1_score_dil,
)

loss = Weighted_Cross_Entropy(10)
precision_dil = Precision_dil
f1_score = F1_score
f1_score_dil = F1_score_dil

model = load_model(
    "models/unet_classification.h5",
    custom_objects={
        "loss": loss,
        "Precision_dil": precision_dil,
        "F1_score": f1_score,
        "F1_score_dil": f1_score_dil,
    },
)



def segmentation(path):
    """
    Generates an overlay image of the input image with the segmented crack

    Args:
        path (str): receives the path of the images for segmentation

    Returns:
        overlay (PIL.Image): overlay image of the input image with the segmented crack
    """

    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    segmented_output = Model(
        inputs=model.input, outputs=model.get_layer(name="sigmoid").output
    )

    y_pred = segmented_output.predict(np.expand_dims(img, axis=0))
    
    # shows the segmented image
    # cv2.imshow('segmented_image', y_pred[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    binary_mask = np.where(y_pred > 0.5, 1, 0)  
    binary_mask = np.squeeze(binary_mask)
    
    color_mask = np.zeros((224, 224, 3), dtype=np.uint8)
    color_mask[binary_mask == 1] = (0, 0, 255)  
   
    img = img.astype('float32')
    color_mask = color_mask.astype('float32')
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay = cv2.resize(overlay, (224, 224))
    
    overlay = Image.fromarray((overlay * 255).astype(np.uint8))
    
    return overlay


def classification(path):
    """Classifies the input image as containing a crack or not

    Args:
        Args:
        path (str): receives the path of the images for classification

    Returns:
        negative (float): probability of the image not containing a crack
        positive (float): probability of the image containing a crack
    """

    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    y_pred = model.predict(img)

    negative = y_pred[0][0] * 100
    positive = y_pred[0][1] * 100

    return (negative, positive)

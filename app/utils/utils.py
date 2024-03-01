import numpy as np
import cv2
import os
from keras.models import load_model
from loss_metrics import *

loss = Weighted_Cross_Entropy(10)
precision_dil = Precision_dil
f1_score = F1_score
f1_score_dil = F1_score_dil

model = load_model(
    'models/unet.h5',
    custom_objects={
        'loss': loss,
        'Precision_dil': precision_dil,
        'F1_score': f1_score,
        'F1_score_dil': f1_score_dil
    }
)

model.load_weights(
    'models/unet.h5'
)


def segmentation(path):
    """
    Generates output predictions for the input samples
    
    Args:
        path (str): receives the path of the images for segmentation

    Returns:
        y_pred (numpy array): array containing the predictions
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    y_pred = model.predict(np.expand_dims(img, axis=0))

    return y_pred

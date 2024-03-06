import numpy as np
import cv2
import os
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
    Generates output predictions for the input samples

    Args:
        path (str): receives the path of the images for segmentation

    Returns:
        y_pred (numpy array): array containing the predictions
    """

    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    segmented_output = Model(
        inputs=model.input, outputs=model.get_layer(name="sigmoid").output
    )

    y_pred = segmented_output.predict(np.expand_dims(img, axis=0))

    return y_pred


def classification(path):
    """Classifies the input image as containing a crack or not

    Args:
        Args:
        path (str): receives the path of the images for classification

    Returns:
        int: 1 if the image contains a crack, 0 otherwise
    """

    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    y_pred = model.predict(img)

    negative = y_pred[0][0] * 100
    positive = y_pred[0][1] * 100

    return (negative, positive)

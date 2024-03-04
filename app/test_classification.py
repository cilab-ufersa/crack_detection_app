import numpy as np
import cv2
from keras.models import load_model

from utils.loss_metrics import (
    F1_score,
    F1_score_dil,
    Precision_dil,
    Weighted_Cross_Entropy,
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

img = cv2.imread("app/temp/")
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0)
img = img / 255.0

y_pred = model.predict(img)
print(np.argmax(y_pred[0]))
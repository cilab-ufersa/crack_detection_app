import numpy as np
import tensorflow as tf
import keras.backend as K
import cv2


def dilation2d(img4D):
    # The greater the kernel size, the greater extent of the dilation applied
    kernel_size = 5

    with tf.compat.v1.variable_scope('dilation2d'):
        kernel = tf.zeros((kernel_size, kernel_size, 1))
        output4D = tf.nn.dilation2d(
            input=img4D,
            filters=kernel,
            strides=(1, 1, 1, 1),
            data_format='NHWC',
            dilations=(1, 1, 1, 1), padding="SAME"
        )

        return output4D


def Weighted_Cross_Entropy(beta):
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(
            logits=y_pred, labels=y_true, pos_weight=beta)

        return tf.reduce_mean(loss)

    return loss

# Focal Loss


def Focal_Loss(gamma=2., alpha=.25):
    """
    Usage:
      model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()

        # Clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
            - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

# F1-score Loss


def F1_score_Loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score

# F1-score Loss with dilated y_true mask


def F1_score_Loss_dil(y_true, y_pred):
    smooth = 1.
    # Dilate y_true
    y_true_dil = dilation2d(y_true)
    y_true_dil = K.flatten(y_true_dil)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_dil * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score

# Recall Metric


def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    return recall

# Precision Metric


def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

# Precision Metric with dilated y_true mask


def Precision_dil(y_true, y_pred):
    # Dilate y_true
    y_true = dilation2d(y_true)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# F1-score Metric


def F1_score(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# F1-score Metric with dilated y_true mask


def F1_score_dil(y_true, y_pred):
    precision = Precision_dil(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Define metrics to be used for evaluation of the trained model using NumPy instead of tensorflow tensors


def DilateMask(mask, threshold=0.5, iterations=1):
    """
    receives mask and returns dilated mask
    """
    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = mask.copy()
    mask_dilated = cv2.dilate(mask_dilated, kernel, iterations=iterations)

    # Binarize mask after dilation
    mask_dilated = np.where(mask_dilated > threshold, 1., 0.)

    return mask_dilated


def Recall_np(y_true, y_pred, threshold=0.5):
    eps = 1e-07
    y_true_f = y_true.flatten().astype('float32')
    half = (np.ones(y_true_f.shape)*threshold).astype('float32')
    y_pred_f = np.greater(y_pred.flatten(), half).astype('float32')
    true_positives = (y_true_f * y_pred_f).sum()
    possible_positives = y_true_f.sum()
    recall = (true_positives + eps) / (possible_positives + eps)

    return recall


def Precision_np(y_true, y_pred, threshold=0.5):
    eps = 1e-07
    y_true_f = y_true.flatten().astype('float32')
    half = (np.ones(y_true_f.shape)*threshold).astype('float32')
    y_pred_f = np.greater(y_pred.flatten(), half).astype('float32')
    true_positives = (y_true_f * y_pred_f).sum()
    predicted_positives = y_pred_f.sum()
    precision = (true_positives + eps) / (predicted_positives + eps)

    return precision


def F1_score_np(recall, precision):
    eps = 1e-07

    return 2*((precision*recall)/(precision+recall+eps))
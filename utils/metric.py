import tensorflow as tf
from tensorflow.keras import backend as K

def recall(y_true,y_pred):
    """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) #逐元素clip（将超出指定范围的数强制变为边界值）
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true,y_pred):
    """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):

# F1 就是这么计算滴
    Precision = precision(y_true,y_pred)
    Recall = recall(y_true, y_pred)
    # F1 = 1-F1
    return 1 - 2 * ((Precision * Recall) / (Precision + Recall + K.epsilon()))


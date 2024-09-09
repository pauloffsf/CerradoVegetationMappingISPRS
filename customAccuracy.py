from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf


'''
Function to calculate the accuracy during training.
This custom accuracy was used because there are several masked pixels in the training data (especially the second level of classification.)
'''

def masked_categoricalAcc():
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.not_equal(class_id_true, 0), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fn
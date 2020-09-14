import tensorflow.keras.backend as K

def F1Score(y_true, y_pred):
    epsilon = K.epsilon()
    
    positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    pred_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    precision = true_positives / (pred_positives + epsilon)
    recall = true_positives / (positives + epsilon)
    
    f1 = 2*(precision * recall) / (precision + recall + epsilon)
    
    return f1

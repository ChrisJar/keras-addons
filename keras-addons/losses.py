import tensorflow as tf
import tensorflow.keras.backend as K

def binary_focal_loss(alpha=0.25, gamma=2.):
    
    def inner(y_true, y_pred):
        epsilon = K.epsilon()
        
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        
        alphas = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alphas, 1. - alphas)
        
        loss = alpha_t * K.pow((1. - p_t), gamma) * -K.log(p_t)
        
        return K.mean(K.sum(loss, axis=1))
    
    return inner

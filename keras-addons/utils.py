import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LambdaCallback
import tempfile

class LearningRateFinder:
    
    def __init__(self, model, stop_factor=4, beta=0.98):
        self.model = model
        self.stop_factor = stop_factor
        self.beta = beta
        
        def find(self, data, epochs=1, start_lr=1e-7, end_lr=10., verbose=1):
        self.lrs = []
        self.losses = []
        self.avg_loss = 0
        self.best_loss = 1e9
        self.batch = 0
        
        its = epochs * len(data)
        self.lr_mult = (end_lr / start_lr) ** (1. / its)
        
        self.weights_file = tempfile.mkstemp()[1]
        self.model.save_weights(self.weights_file)
        
        orig_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, start_lr)
        
        cb = LambdaCallback(on_batch_end=lambda batch,logs: self.on_batch_end(batch, logs))
        
        self.model.fit(
            x=data,
            epochs=epochs,
            verbose=verbose,
            callbacks=[cb]
        )
        
        self.model.load_weights(self.weights_file)
        K.set_value(self.model.optimizer.lr, orig_lr)
        
    def on_batch_end(self, batch, logs):
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        
        loss = logs["loss"]
        self.batch += 1
        self.avg_loss = (self.beta * self.avg_loss) + ((1 - self.beta) * loss)
        smooth = self.avg_loss / (1 - (self.beta ** self.batch))
        self.losses.append(smooth)
        
        if self.batch > 1 and smooth > self.stop_factor * self.best_loss:
            self.model.stop_training = True
            return
        
        if self.batch == 1 or smooth < self.best_loss:
            self.best_loss = smooth
            
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)
        
        
    def plot_loss(self, skip_begin=10, skip_end=1):
        lrs = self.lrs[skip_begin:-skip_end]
        losses = self.losses[skip_begin:-skip_end]
        
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate vs Loss")
        
 

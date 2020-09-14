import tensorflow.keras.backend as K
import math

class OneCycleScheduler(Callback):
    
    def __init__(self, start_lr=1e-4, max_lr=3e-3, moms=None, switch_point=0.3):
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.switch_point = switch_point
        self.iteration = 0
        
        if moms:
            self.cycle_moms = True
            self.max_mom = moms[0]
            self.min_mom = moms[1]
        else:
            self.cycle_moms = False
    
    
    def on_train_begin(self, logs=None):
        self.n = self.params['steps'] * self.params['epochs']
        self.p1 = int(self.n * self.switch_point)
        self.p2 = self.n-self.p1
        
        K.set_value(self.model.optimizer.lr, self.start_lr)
        
        if self.cycle_moms:
            K.set_value(self.model.optimizer.momentum, self.max_mom)
    
    def on_train_batch_end(self, batch, logs=None):
        K.set_value(self.model.optimizer.lr, self.lr_sched())
        
        if self.cycle_moms:
            K.set_value(self.model.optimizer.momentum, self.mom_sched())
        
        self.iteration += 1
    
    def lr_sched(self):
        i = self.iteration
        p1 = self.p1
        p2 = self.p2
        
        if i <= p1:
            pos = i / p1
            return self.cos_sched(self.start_lr, self.max_lr, pos)
        else:
            pos = (i-p1) / p2
            return self.cos_sched(self.max_lr, 0., pos)
    
    def mom_sched(self):
        i = self.iteration
        p1 = self.p1
        p2 = self.p2
        
        if i <= p1:
            pos = i / p1
            return self.cos_sched(self.max_mom, self.min_mom, pos)
        else:
            pos = (i-p1) / p2
            return self.cos_sched(self.min_mom, self.max_mom, pos)
    
    def cos_sched(self, start, end, pos):
        return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
        

'''
Custom callbacks
'''
import math
from tensorflow.keras.callbacks import Callback
class ValidationCallback(Callback):
    '''
    Custom callback to print validation accuracy every 'nth' epoch
    '''
    def __init__(self, validation_data_steps):
        self.validation_data, self.steps = validation_data_steps
    def on_epoch_end(self, epoch, logs={}):
        '''
         Callback method to print validation accuracy after every 5th epoch
        '''
        # if (epoch > 1) and ((epoch - 1) % 5 == 0):
        self.model.batch_size = 2
        loss, acc = self.model.evaluate(self.validation_data, steps=self.steps, verbose=1)
        print('\nValidation loss: {0}, acc: {1}\n'.format(loss, acc))

def step_decay(epoch):
    '''
    Drop learning rate by some percent after every 'nth' epoch
    '''
    initial_lrate = 1e-4
    drop = 0.1
    epochs_drop = 25.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    lrate = round(lrate, math.floor((1+epoch)/epochs_drop) + 4)
    return lrate

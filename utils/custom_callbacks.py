'''
This module contains custom callbacks that can be imported while training the network
'''
import math
from tensorflow.keras.callbacks import Callback
def step_decay(epoch):
    '''
    Description
    -----------
    Decrease learning rate by a certain percentage after a specified interval

    Args
    ----
    epoch: latest training epoch number
    '''
    #Should be equal to the inital learning rate
    initial_lrate = 1e-4
    #Interval after which learning rate must be dropped
    interval = 25.0
    #Percentage to drop the learning rate by
    drop = 0.1
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/interval))
    lrate = round(lrate, math.floor((1+epoch)/interval) + 4)
    return lrate

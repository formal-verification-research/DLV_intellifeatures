#!/usr/bin/env python

from keras.models import model_from_json
import sys
sys.path.append('networks')
sys.path.append('safety_check')
sys.path.append('adversary_generation')
sys.path.append('configuration')

import os
import time
import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt
# from pylab import *

# keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import keras.optimizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

# visualisation
# from keras.utils.visualize_util import plot
#
from keras.datasets import mnist
from keras.utils import np_utils

# for training cifar10
from keras.preprocessing.image import ImageDataGenerator




#from configuration import *


print("Start loading model ... ")
weightFile = '/home/souravsanyal06/DLV_intellifeatures/networks/mnist/mnist.mat'
modelFile = '/home/souravsanyal06/DLV_intellifeatures/networks/mnist/mnist.json'

weights = sio.loadmat(weightFile)
model = model_from_json(open(modelFile).read())

model.save('/home/souravsanyal06/DLV_intellifeatures/networks/mnist//mnist_r.hd5', overwrite=True)



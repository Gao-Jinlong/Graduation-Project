import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from nets.unet import Unet
from nets.unet_training import(CE, Focal_loss, dice_loss_with_CE)
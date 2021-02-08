import collections
import functools
import gc
import numpy as np
from pathlib import Path
import operator
import os
import sonnet as snt
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import tensorflow.compat.v1 as tf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from torch.utils.tensorboard import SummaryWriter

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# tfc = tf.compat.v1
device = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device)

print(device)
print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'))
time.sleep(300)
print('DONE')

with open('/results/DONE', 'w') as f:
    f.write('voila')

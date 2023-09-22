import os
import gc
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
testtest
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
gpus = tf.config.list_physical_devices('GPU')

from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
import numpy as np


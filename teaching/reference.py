import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import tensorflow as tf
import gc

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')

def ReshapeWeight(weight_record):
    weight_reshape = []
    for weights in weight_record:
        weight_tmp = []
        for weight in weights:
            weight = np.reshape(weight, (-1,))
            weight_tmp = np.hstack((weight_tmp, weight))
        if weight_reshape == []:
            weight_reshape = weight_tmp
        else:
            weight_reshape = np.vstack((weight_reshape, weight_tmp))

# load MNIST data.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Preprocessing.
x_train = x_train / 255.0
x_test = x_test / 255.0
shuffle_index = np.random.permutation(len(x_train))
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
# Track the data type.
dataType = x_train.dtype
print(f"Data type: {dataType}")
labelType = y_test.dtype
print(f"Data type: {labelType}")

x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Create the model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model_ref = Sequential()
model_ref.add(Dense(units=32, activation='relu'))
model_ref.add(Dense(units=32, activation='relu'))
model_ref.add(Dense(units=10, activation='softmax'))

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
# Instantiate a loss function.
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

epochs = 3000
weight_record = []
for epoch in range(epochs):
    # Iterate over the batches of the dataset.
    with tf.GradientTape() as tape:
        logits = model_ref(x_train, training = True)
        loss_value = loss_fn(y_train, logits)
    grads = tape.gradient(loss_value, model_ref.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_ref.trainable_weights))
    cur_weight = []
    for weight in model_ref.trainable_weights:
        cur_weight.append(weight.numpy())
    weight_record.append(cur_weight)
    if epoch % 50 == 0:
        print("Training loss at epoch %d: %.4f" % (epoch, float(loss_value)))
        weight_record = ReshapeWeight(weight_record)
        np.save('ref_output/weight_ref_%d' %(epoch/50),weight_record)
        
        del weight_record
        gc.collect()
        weight_record = []

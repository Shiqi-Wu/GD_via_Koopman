import os
import gc
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
gpus = tf.config.list_physical_devices('GPU')

from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf

# Load MNIST data.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessing: normalize pixel values to be between 0 and 1.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Shuffle training data.
shuffle_index = np.random.permutation(len(x_train))
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

# Convert target labels to binary classification (digit < 5 or digit >= 5).
y_train = (y_train < 5)
y_test = (y_test < 5)

# Convert labels to one-hot encoding.
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Track the data type of x_train.
dataType = x_train.dtype
print(f"Data type: {dataType}")

# Track the data type of y_test.
labelType = y_test.dtype
print(f"Data type: {labelType}")

# Print the one-hot encoded label for the first training example.
print(y_train[0])

# Instantiate the training dataset.
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Perform PCA to reduce dimensionality of x_train.
pca = PCA(n_components=0.7)  # retain 95% of variance
x_train_reduced = pca.fit_transform(x_train.reshape(x_train.shape[0], -1))
x_test_reduced = pca.transform(x_test.reshape(x_test.shape[0], -1))

# Print the dimensions of the original and reduced datasets.
print("Original dimensions:", x_train.shape)
print("Reduced dimensions:", x_train_reduced.shape)

# node_set = [16, 32, 64, 128, 256, 512, 1024]
# node_set = [32, 64]
node_set = [16]



from tensorflow import keras

folder_path = os.path.join(os.getcwd(), '../learn_output')

for node_num in node_set:
    for training_id in range(171,400):
        inputs = keras.Input(shape=(26,), name="digits")
        x = keras.layers.Dense(node_num, activation="relu", kernel_initializer="uniform",bias_initializer="uniform")(inputs)
        outputs = keras.layers.Dense(2, name="predictions",kernel_initializer="uniform",bias_initializer="uniform")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Instantiate an optimizer.
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        # Instantiate a loss function.
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        # Compile the model
        model.compile(optimizer=optimizer, loss=loss_fn,  metrics=["accuracy"])        
        
        weight_record = []
        print("Start training at width %d, id %d" % (node_num, training_id))
        for epoch in range(50):
            history = model.fit(x_train_reduced, y_train, validation_data=(x_test_reduced, y_test), batch_size=128, epochs=1, verbose=2)
            cur_weights = [w.numpy() for w in model.trainable_weights]
            # print(cur_weights)
            cur_weights = [w.flatten() for w in cur_weights]
            cur_weights = np.concatenate(cur_weights)
            cur_weights = cur_weights.reshape(-1)
            weight_record.append(cur_weights)
            # print(cur_weights)
        
        # print(weight_record)
        print(np.shape(weight_record))
        file_path = os.path.join(folder_path, 'weight50_%d_%d.npy' % (node_num, training_id))
        np.save(file_path, weight_record)
        del weight_record
        gc.collect()
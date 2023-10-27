node_num = 512
latent_num = 32
suffix = '_512node_32latent_1step'

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # logical_gpus = tf.config.list_logical_devices('GPU')
    # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

import numpy as np

training_id = 0
check_point = 350
end_point = 400

folder_path = os.path.join(os.getcwd(), '../learn_output')
file_path = os.path.join(folder_path, 'weight50_%d_%d.npy' % (node_num, training_id))

weight = np.load(file_path, allow_pickle=True)

x_train_para = weight[:-1,:]
y_train_para = weight[1:,:]

for training_id in range(1, check_point):
    file_path = os.path.join(folder_path, 'weight50_%d_%d.npy' % (node_num, training_id))
    
    weight = np.load(file_path, allow_pickle=True)
    
    x_train_para = np.concatenate((x_train_para, weight[:-1,:]), axis = 0)
    y_train_para = np.concatenate((y_train_para, weight[1:,:]), axis = 0)

training_id = check_point
file_path = os.path.join(folder_path, 'weight50_%d_%d.npy' % (node_num, training_id))
weight = np.load(file_path, allow_pickle=True)

x_test_para = weight[:-1,:]
y_test_para = weight[1:,:]

for training_id in range(check_point + 1, end_point):
    file_path = os.path.join(folder_path, 'weight50_%d_%d.npy' % (node_num, training_id))
    
    weight = np.load(file_path, allow_pickle=True)
    
    x_test_para = np.concatenate((x_test_para, weight[:-1,:]), axis = 0)
    y_test_para = np.concatenate((y_test_para, weight[1:,:]), axis = 0)


import matplotlib.pyplot as plt
import seaborn as sns

permuted_indices = np.random.permutation(x_train_para.shape[0])
x_train_para = x_train_para[permuted_indices, :]
y_train_para = y_train_para[permuted_indices, :]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train_para)
x_train_para_scaled = scaler.transform(x_train_para)
y_train_para_scaled = scaler.transform(y_train_para)
x_test_para_scaled = scaler.transform(x_test_para)
y_test_para_scaled = scaler.transform(y_test_para)



from sklearn.decomposition import PCA

# Load MNIST data.
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

# Preprocessing: normalize pixel values to be between 0 and 1.
x_train_mnist = x_train_mnist / 255.0
x_test_mnist = x_test_mnist / 255.0

# Shuffle training data.
shuffle_index = np.random.permutation(len(x_train_mnist))
x_train_mnist, y_train_mnist = x_train_mnist[shuffle_index], y_train_mnist[shuffle_index]

# Convert target labels to binary classification (digit < 5 or digit >= 5).
y_train_mnist = (y_train_mnist < 5)
y_test_mnist = (y_test_mnist < 5)

# Convert labels to one-hot encoding.
y_train_mnist = tf.keras.utils.to_categorical(y_train_mnist)
y_test_mnist = tf.keras.utils.to_categorical(y_test_mnist)

# Instantiate the training dataset.
x_train_mnist = np.reshape(x_train_mnist, (-1, 784))
x_test_mnist = np.reshape(x_test_mnist, (-1, 784))

# Perform PCA to reduce dimensionality of x_train.
pca = PCA(n_components=0.7)  # retain 70% of variance
x_train_reduced_mnist = pca.fit_transform(x_train_mnist.reshape(x_train_mnist.shape[0], -1))
x_test_reduced_mnist = pca.transform(x_test_mnist.reshape(x_test_mnist.shape[0], -1))


x_combined_reduced_mnist = np.concatenate((x_train_reduced_mnist, x_test_reduced_mnist), axis=0)
y_combined_reduced_mnist = np.concatenate((y_train_mnist, y_test_mnist), axis=0)


from tensorflow import keras

inputs = keras.Input(shape=(26,), name="digits")
x = keras.layers.Dense(node_num, activation="relu", kernel_initializer="uniform",bias_initializer="uniform")(inputs)
outputs = keras.layers.Dense(2, name="predictions",kernel_initializer="uniform",bias_initializer="uniform")(x)
model_mnist = keras.Model(inputs=inputs, outputs=outputs, name = 'model_mnist')

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
# Instantiate a loss function.
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# Compile the model
model_mnist.compile(optimizer=optimizer, loss=loss_fn,  metrics=["accuracy"])



model_mnist.summary()


# ## Parameter Model

# Build the model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.layers import Input, Add, Multiply, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.executing_eagerly()

class DicNN(Layer):
    """
    Trainable disctionries
    """
    
    def __init__(self, n_input, layer_sizes=[64, 64], n_psi_train=64, **kwargs):
        """_summary_
        Args:
            layer_sizes (list, optional): Number of unit of hidden layer, activation = 'tanh'. Defaults to [64, 64].
            n_psi_train (int, optional): Number of unit of output layer. Defaults to 22.
        """
        super(DicNN, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.input_layer = Dense(self.layer_sizes[0], name='Dic_input', use_bias=False)
        self.hidden_layers = [Dense(layer_sizes[i], activation='tanh', name='Dic_hidden_%d'%i) for i in range(len(layer_sizes))]        
        self.output_layer = Dense(n_psi_train, name='Dic_output')
        self.n_psi_train = n_psi_train
        self.inv_input_layer = Dense(self.layer_sizes[-1], name = 'Dic_input_inv', use_bias=False)
        self.inv_hidden_layers = [Dense(layer_sizes[-(i+1)], activation='tanh', name='Dic_hidden_%d_inv'%i) for i in range(len(layer_sizes))]
        self.inv_output_layer = Dense(n_input, name = 'Dic_output_inv')
        self.n_input = n_input
        
    def call(self, inputs):
        psi_x_train = self.input_layer(inputs)
        for layer in self.hidden_layers:
            psi_x_train = psi_x_train + layer(psi_x_train)
        outputs = self.output_layer(psi_x_train)
        return outputs
    
    def inv_call(self, inputs):
        x_inv = self.inv_input_layer(inputs)
        for layer in self.inv_hidden_layers:
            x_inv = x_inv + layer(x_inv)
        outputs = self.inv_output_layer(x_inv)
        return outputs
    
    def get_config(self):
        config = super(DicNN, self).get_config()
        config.update({
            'layer_sizes': self.layer_sizes,
            'n_psi_train': self.n_psi_train
        })
        return config



import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer, Subtract
from tensorflow.keras.models import Model

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, weights_shape, biases_shape, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.weights_shape = weights_shape
        self.biases_shape = biases_shape

    def call(self, x, theta_w, theta_b, m):
        x = tf.cast(x, dtype=tf.float32)
        w = tf.reshape(theta_w, self.weights_shape)
        b = tf.reshape(theta_b, self.biases_shape)
        b_expanded = tf.tile(tf.expand_dims(b, axis=1), [1, m, 1])
        y = tf.matmul(x, w) + b_expanded
        return y

class ModelLayer(tf.keras.layers.Layer):
    def __init__(self, model, data, **kwargs):
        super(ModelLayer, self).__init__(**kwargs)
        self.model = model
        self.data = data
        self.layer_shape = []
        self.layer_size = []
        self.m = np.shape(data)[0]
        for layer in self.model.layers:
            weights = layer.get_weights()
            if len(weights) > 0:
                weights_shape = tf.shape(weights[0])
                biases_shape = tf.shape(weights[1])
                target_shapes = [weights_shape, biases_shape]
                target_shapes = [tf.concat([[-1], shape], axis=0) for shape in target_shapes]
                weights_size = tf.size(weights[0]).numpy()
                biases_size = tf.size(weights[1]).numpy()
                self.layer_shape.append(target_shapes)
                self.layer_size.append([weights_size, biases_size])            
            print(self.layer_shape)
            print(self.layer_size)
        
        self.Layers = []
        for shape in self.layer_shape:
            if shape != None:
                self.Layers.append(CustomLayer(shape[0], shape[1]))
                
    def call(self, theta_M):
        index = 0
        y = self.data
                
        # Hidden Layers
        for i in range(len(self.layer_shape) - 1):
#             print(index)
            size = self.layer_size[i]
            theta_w = theta_M[:, index:index + size[0]]
            theta_b = theta_M[:, index + size[0]: index + size[0] + size[1]]
            index += size[0] + size[1]
            y = self.Layers[i].call(y, theta_w, theta_b, self.m)
            y = tf.nn.relu(y)
        
        # Output Layer
        size = self.layer_size[-1]
        theta_w = theta_M[:, index:index + size[0]]
        theta_b = theta_M[:, index + size[0]: index + size[0] + size[1]]
        y = self.Layers[-1].call(y, theta_w, theta_b, self.m)
        return y


# layer_f
Layer_f = ModelLayer(model_mnist, x_train_reduced_mnist)


def Build_model(x_train_para = x_train_para, n_psi_train = latent_num):
    # model_psi
    dic = DicNN(n_input = np.shape(x_train_para)[1], n_psi_train = n_psi_train)
    inputs_x = Input((np.shape(x_train_para)[1],))
    model_psi = Model(inputs = inputs_x, outputs = dic.call(inputs_x), name = 'model_psi')
    
    # model_koopman
    inputs_x = Input((np.shape(x_train_para)[1],))
    psi_x = model_psi(inputs_x)
    k_layer = Dense(units = dic.n_psi_train, use_bias=False, name = 'k_layer')
    outputs_x = k_layer(psi_x)
    inputs_y = Input((np.shape(x_train_para)[1],))
    psi_y = model_psi(inputs_y)
    outputs = outputs_x - psi_y
    model_koopman = Model(inputs = [inputs_x, inputs_y], outputs = outputs, name = 'model_koopman')
    
    # model_inverse
    inputs_kpsi = Input((dic.n_psi_train,))
    model_inv_psi = Model(inputs = inputs_kpsi, outputs = dic.inv_call(inputs_kpsi), name = 'model_psi_inv')
    
    # model_predict
    inputs_x = Input((np.shape(x_train_para)[1],))
    psi_x = model_psi(inputs_x)
    psi_x_predict = k_layer(psi_x)
    outputs_predict = model_inv_psi(psi_x_predict)
    model_predict = Model(inputs = inputs_x, outputs = outputs_predict, name = 'model_predict')
    
    # autoencoder
    inputs_x = tf.keras.layers.Input(shape=(x_train_para.shape[1],))
    psi_x = model_psi(inputs_x)
    x_hat = model_inv_psi(psi_x)
    model_auto = Model(inputs = inputs_x, outputs = x_hat, name = 'model_auto')

    return model_psi, model_koopman, model_inv_psi, model_predict, model_auto


model_psi, model_koopman, model_inv_psi, model_predict, model_auto = Build_model(x_train_para = x_train_para, n_psi_train = latent_num)


model_predict.summary()


# Create a train_dataset using from_tensor_slices
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_para, y_train_para))
val_dataset = tf.data.Dataset.from_tensor_slices((x_test_para, y_test_para))


# Shuffle and batch the dataset (you can adjust batch_size and buffer_size as needed)
batch_size = 16
buffer_size = 128
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size)


# Optionally, you can prefetch data for faster training
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Define the optimizer

# initial_learning_rate = 0.001
# lr_schedule = ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=30, 
#     decay_rate=0.2
# )

optimizer = Adam(learning_rate=0.0001)

# Define the Mean Squared Error loss function
def MeanSquaredError():
    return tf.keras.losses.MeanSquaredError()

# Define the loss function (as you've defined previously)
lambda_1 = 10.0
lambda_2 = 0.1
lambda_3 = 0.1

# Get the list of trainable variables
trainable_variables = model_koopman.trainable_variables + model_predict.trainable_variables + model_auto.trainable_variables

# Define a function to compute and apply gradients
clip_norm_value = 1.0

@tf.function
def train_step(x_data, y_data):
    with tf.GradientTape() as tape:
        # Calculate the loss
        koopman_loss = MeanSquaredError()(tf.zeros_like(model_psi(x_data)), model_koopman([x_data, y_data]))
        predicted_y = Layer_f(model_predict(x_data))
        reconstruction_loss = MeanSquaredError()(predicted_y, Layer_f(y_data))
        autoencoder_loss = MeanSquaredError()(Layer_f(model_auto(x_data)), Layer_f(x_data))
        total_loss = lambda_1 * koopman_loss + lambda_2 * reconstruction_loss + lambda_3 * autoencoder_loss

    # Compute gradients
    gradients = tape.gradient(total_loss, trainable_variables)

    # Apply gradient clipping
    clipped_gradients = [tf.clip_by_norm(grad, clip_norm_value) for grad in gradients]

    # Apply gradients to update model parameters
    optimizer.apply_gradients(zip(clipped_gradients, trainable_variables))
    
    return total_loss, koopman_loss, reconstruction_loss, autoencoder_loss

# Define a function to compute loss on validation data
@tf.function
def validate_step(x_data, y_data):
    # Calculate the loss
    koopman_loss = MeanSquaredError()(tf.zeros_like(model_psi(x_data)), model_koopman([x_data, y_data]))
    predicted_y = Layer_f(model_predict(x_data))
    reconstruction_loss = MeanSquaredError()(predicted_y, Layer_f(y_data))
    autoencoder_loss = MeanSquaredError()(Layer_f(model_auto(x_data)), Layer_f(x_data))
    total_loss = lambda_1 * koopman_loss + lambda_2 * reconstruction_loss + lambda_3 * autoencoder_loss
    
    return total_loss, koopman_loss, reconstruction_loss, autoencoder_loss

# Initialize lists to store training and validation losses for each component
train_koopman_losses_per_epoch = []
train_autoencoder_losses_per_epoch = []
train_reconstruction_losses_per_epoch = []
val_koopman_losses_per_epoch = []
val_autoencoder_losses_per_epoch = []
val_reconstruction_losses_per_epoch = []
train_losses_per_epoch = []
val_losses_per_epoch = []

# Perform validation after each training epoch
num_epochs = 100
for epoch in range(num_epochs):
    total_loss_epoch = 0.0
    
    # Initialize epoch-specific loss variables for each component
    koopman_loss_epoch = 0.0
    autoencoder_loss_epoch = 0.0
    reconstruction_loss_epoch = 0.0
    
    # Training loop (similar to your code)
    for batch_x_data, batch_y_data in train_dataset:
        loss, koopman_loss, reconstruction_loss, autoencoder_loss = train_step(batch_x_data, batch_y_data)
        total_loss_epoch += loss.numpy()
        
        # Accumulate losses for each component
        koopman_loss_epoch += koopman_loss.numpy()
        autoencoder_loss_epoch += autoencoder_loss.numpy()
        reconstruction_loss_epoch += reconstruction_loss.numpy()
    
    # Calculate the average training loss for the epoch (similar to your code)
    avg_loss_epoch = total_loss_epoch / len(train_dataset)
    
    # Validation loop
    total_val_loss_epoch = 0.0
    koopman_val_loss_epoch = 0.0
    autoencoder_val_loss_epoch = 0.0
    reconstruction_val_loss_epoch = 0.0
    
    for batch_x_val_data, batch_y_val_data in val_dataset:
        val_loss, koopman_val_loss, reconstruction_val_loss, autoencoder_val_loss = validate_step(batch_x_val_data, batch_y_val_data)
        total_val_loss_epoch += val_loss.numpy()
        
        # Accumulate validation losses for each component
        koopman_val_loss_epoch += koopman_val_loss.numpy()
        autoencoder_val_loss_epoch += autoencoder_val_loss.numpy()
        reconstruction_val_loss_epoch += reconstruction_val_loss.numpy()
    
    # Calculate the average validation loss for the epoch
    avg_val_loss_epoch = total_val_loss_epoch / len(val_dataset)
    
    # Append the training and validation losses for each component to their respective lists
    train_koopman_losses_per_epoch.append(koopman_loss_epoch / len(train_dataset))
    train_autoencoder_losses_per_epoch.append(autoencoder_loss_epoch / len(train_dataset))
    train_reconstruction_losses_per_epoch.append(reconstruction_loss_epoch / len(train_dataset))
    
    val_koopman_losses_per_epoch.append(koopman_val_loss_epoch / len(val_dataset))
    val_autoencoder_losses_per_epoch.append(autoencoder_val_loss_epoch / len(val_dataset))
    val_reconstruction_losses_per_epoch.append(reconstruction_val_loss_epoch / len(val_dataset))
    
    train_losses_per_epoch.append(avg_loss_epoch)
    val_losses_per_epoch.append(avg_val_loss_epoch)

    # Print both training and validation loss for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Train Loss: {avg_loss_epoch}, Val Loss: {avg_val_loss_epoch}, "
          f"Train Koopman Loss: {train_koopman_losses_per_epoch[-1]}, Val Koopman Loss: {val_koopman_losses_per_epoch[-1]}, "
          f"Train Autoencoder Loss: {train_autoencoder_losses_per_epoch[-1]}, Val Autoencoder Loss: {val_autoencoder_losses_per_epoch[-1]}, "
          f"Train Reconstruction Loss: {train_reconstruction_losses_per_epoch[-1]}, Val Reconstruction Loss: {val_reconstruction_losses_per_epoch[-1]}")

# Once training is complete, your model is trained, and you have lists for each component's training and validation losses.


import matplotlib.pyplot as plt

# Assuming you have populated the lists with your data

# Define the x-axis values (epochs)
epochs = range(1, len(train_koopman_losses_per_epoch) + 1)

# Plot the training and validation losses for each component
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_koopman_losses_per_epoch, label='Train Koopman Loss')
plt.plot(epochs, val_koopman_losses_per_epoch, label='Val Koopman Loss')
plt.title('Koopman Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')  # Set y-axis to log scale
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, train_autoencoder_losses_per_epoch, label='Train Autoencoder Loss')
plt.plot(epochs, val_autoencoder_losses_per_epoch, label='Val Autoencoder Loss')
plt.title('Autoencoder Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')  # Set y-axis to log scale
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, train_reconstruction_losses_per_epoch, label='Train Reconstruction Loss')
plt.plot(epochs, val_reconstruction_losses_per_epoch, label='Val Reconstruction Loss')
plt.title('Reconstruction Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')  # Set y-axis to log scale
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, train_losses_per_epoch, label='Train Total Loss')
plt.plot(epochs, val_losses_per_epoch, label='Val Total Loss')
plt.title('Total Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')  # Set y-axis to log scale
plt.legend()

plt.tight_layout()
figure_filename = 'plot/training_loss' + suffix + '.png'

plt.savefig(figure_filename)


# Specify the path where you want to save the weights
weights_filename = 'output/model_parameter' + suffix + '.h5'

# Save the weights to the specified file
model_predict.save_weights(weights_filename)

print("Model weights saved to", weights_filename)



ref_para_test = []
ref_initial_test = []

for training_id in range(check_point, end_point):
    file_path = os.path.join(folder_path, 'weight50_%d_%d.npy' % (node_num, training_id))
    
    weight = np.load(file_path, allow_pickle=True)
    
    ref_para_test.append(weight)
    ref_initial_test.append(weight[0:1,:])


loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

k_layer = model_koopman.get_layer('k_layer')

weights = k_layer.get_weights()

if len(weights) > 0:
    weight_matrix = weights[0]  # 假设权重矩阵在列表中的第一个元素

    eigenvalues = np.linalg.eigvals(weight_matrix)

    eigenvalues_filename = 'output/eigenvalues' + suffix + '.npy'
    np.save(eigenvalues_filename, eigenvalues)



model_inv_psi.summary()


steps_num = 50
loss_ref_set = []
loss_predict_set = []
for j in range(len(ref_para_test)):
    ref_para = ref_para_test[j]
    predict_traj = [ref_initial_test[j]]
    loss_ref = []
    loss_predict = []
    x = predict_traj[-1]
    psi_x = model_psi(x)
    for step in range(steps_num):
        psi_y = k_layer(psi_x)
        psi_x = psi_y
        loss_ref.append(loss_fn(y_train_mnist, Layer_f(ref_para[step:step+1, :])[0,:,:]))
        loss_predict.append(loss_fn(y_train_mnist, Layer_f(predict_traj[-1])[0,:,:]))
        y = model_inv_psi(psi_y)
        predict_traj.append(y)
    loss_ref_set.append(loss_ref)
    loss_predict_set.append(loss_predict)


# Calculate the average and standard deviation of loss for each step
average_loss_ref = np.mean(loss_ref_set, axis=0)
std_loss_ref = np.std(loss_ref_set, axis=0)
average_loss_predict = np.mean(loss_predict_set, axis=0)
std_loss_predict = np.std(loss_predict_set, axis=0)

# Create an array for the steps
steps = np.arange(steps_num)

plt.figure()
# Plot the average loss with error bars
plt.errorbar(steps, average_loss_ref, yerr=std_loss_ref, label='Loss (Reference)', capsize=4, linestyle='--', marker='x')
plt.errorbar(steps, average_loss_predict, yerr=std_loss_predict, label='Loss (Predict)', capsize=4, linestyle='-', marker='.')

# Customize the plot
plt.title('Average Loss with Error Bars')
plt.xlabel('Steps')
plt.ylabel('Average Loss')
plt.yscale('log')
plt.legend()
# plt.grid(True)

# Show the plot
figure_filename = 'plot/mnist_loss' + suffix + '_log.png'
plt.savefig(figure_filename)

import numpy as np

average_loss_ref_filename = 'output/average_loss_ref' + suffix + '.npy'
std_loss_ref_filename = 'output/std_loss_ref' + suffix + '.npy'
average_loss_predict_filename = 'output/average_loss_predict' + suffix + '.npy'
std_loss_predict_filename = 'output/std_loss_predict' + suffix + '.npy'

np.save(average_loss_ref_filename, average_loss_ref)
np.save(std_loss_ref_filename, std_loss_ref)
np.save(average_loss_predict_filename, average_loss_predict)
np.save(std_loss_predict_filename, std_loss_predict)

print("Arrays saved to files with the suffix:", suffix)




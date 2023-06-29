import numpy as np
import scipy
from scipy.sparse import random

m = 100  # Number of rows
n = 100  # Number of columns

# Generate a random sparse matrix
density = 0.3  # Density of non-zero elements
sparse_matrix = random(m, n, density=density, format='csr')

print(sparse_matrix)

def matrix_factorization(A, d, num_iterations, learning_rate):
    m, n = A.shape
    U_history = []
    V_history = []
    loss_history = []


    # Initialize U and V with random values
    U = np.random.rand(m, d)
    V = np.random.rand(n, d)
    U_history.append(U.reshape((1,-1)))
    V_history.append(V.reshape((1,-1)))

    for iteration in range(num_iterations):
        # Compute the error matrix
        error = A - np.dot(U, V.T)
        # Compute the gradients for U and V
        grad_U = -2 * np.dot(error, V)
        grad_V = -2 * np.dot(error.T, U)

        # Update U and V using gradient descent
        U -= learning_rate * grad_U
        V -= learning_rate * grad_V
        
        loss_history.append(np.linalg.norm(error)/(m * n))
        U_history.append(U.reshape((1,-1)))
        V_history.append(V.reshape((1,-1)))
#         print('the error of iteration %d is %f' % (iteration, loss_history[-1]))

    return U, V, U_history, V_history, loss_history

A = scipy.sparse.csr_matrix(sparse_matrix)
d = 10
num_iterations = 100
learning_rate = 0.001


U, V, U_history, V_history, loss_history = matrix_factorization(A, d, num_iterations, learning_rate)

U_history = np.array(U_history)
print(np.shape(U_history))
U_history_x = np.reshape(U_history[:-1], (num_iterations, -1))
U_history_y = np.reshape(U_history[1:], (num_iterations, -1))
V_history = np.array(V_history)
V_history_x = np.reshape(V_history[:-1], (num_iterations, -1))
V_history_y = np.reshape(V_history[1:], (num_iterations, -1))

U_data_x, U_data_y, V_data_x, V_data_y = [U_history_x], [U_history_y], [V_history_x], [V_history_y]

training_num = 1000
for i in range(1, training_num):
    U, V, U_history, V_history, loss_history = matrix_factorization(A, d, num_iterations, learning_rate)
    U_history = np.array(U_history)
    U_history_x = np.reshape(U_history[:-1], (num_iterations, -1))
    U_history_y = np.reshape(U_history[1:], (num_iterations, -1))
    V_history = np.array(V_history)
    V_history_x = np.reshape(V_history[:-1], (num_iterations, -1))
    V_history_y = np.reshape(V_history[1:], (num_iterations, -1))
    U_data_x.append(U_history_x)
    U_data_y.append(U_history_y)
    V_data_x.append(V_history_x)
    V_data_y.append(V_history_y)

U_data_x = np.concatenate(U_data_x, axis=0)
U_data_y = np.concatenate(U_data_y, axis=0)
V_data_x = np.concatenate(V_data_x, axis=0)
V_data_y = np.concatenate(V_data_y, axis=0)

para_data_x = np.concatenate((U_data_x, V_data_x), axis = 1)
para_data_y = np.concatenate((U_data_y, V_data_y), axis = 1)

A_dense = A.toarray()

# Build the model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.layers import Input, Add, Multiply, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import tensorflow as tf
from tensorflow import keras

class DicNN(Layer):
    """
    Trainable disctionries
    """
    
    def __init__(self, n_input, layer_sizes=[128,128,128], n_psi_train=64, **kwargs):
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

dic = DicNN(n_input = np.shape(para_data_x)[1], n_psi_train = 106)
inputs_x = Input((np.shape(para_data_x)[1],))
model_psi = Model(inputs = inputs_x, outputs = dic.call(inputs_x), name = 'model_psi')


def matrix_residual(para_data, A, shape):
    para_reshape = tf.reshape(para_data, (-1, shape[0], shape[1], shape[2]))
    U = para_reshape[:,0,:,:]
    V = para_reshape[:,1,:,:]
    V_T = tf.transpose(V, perm=[0, 2, 1])
    result = tf.matmul(U, V_T)
    result_minus_A = tf.subtract(tf.cast(result, tf.float64), tf.tile(tf.expand_dims(A, 0), [tf.shape(para_data)[0], 1, 1]))
    return tf.reshape(result_minus_A, (tf.shape(para_data)[0],-1))

shape = (2,100,10)

inputs_x = Input((np.shape(para_data_x)[1],))
inputs_y = Input((np.shape(para_data_x)[1],))
psi_x = model_psi(inputs_x)
psi_y = model_psi(inputs_y)
k_layer = Dense(units = dic.n_psi_train, use_bias=False, name = 'k_layer')
outputs_x = k_layer(psi_x)
outputs = outputs_x - psi_y
model_koopman = Model(inputs = [inputs_x, inputs_y], outputs = outputs, name = 'model_koopman')


# In[41]:


inputs_x = Input((np.shape(para_data_x)[1],))
psi_x = model_psi(inputs_x)
inputs_kpsi = Input((dic.n_psi_train,))
model_inv_psi = Model(inputs = inputs_kpsi, outputs = dic.inv_call(inputs_kpsi))
model_auto = Model(inputs = inputs_x, outputs = model_inv_psi(psi_x), name = 'model_auto')

input_x = tf.keras.layers.Input(shape=(para_data_x.shape[1],))
input_y = tf.keras.layers.Input(shape=(para_data_x.shape[1],))
output_auto = model_auto(input_x)
output_koopman = model_koopman([input_x, input_y])
output_error = matrix_residual(output_auto, A_dense, shape)
model_error = tf.keras.models.Model(inputs=input_x, outputs=output_error, name = 'model_error')
combined_model = tf.keras.models.Model(inputs=[input_x, input_y], outputs=[model_error(input_x), output_koopman], name = 'model_combined')

batch_size = 4096
epochs = 1000
lbd = 0.1

combined_model.compile(optimizer='adam', loss='mse')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

para_data_x_train, para_data_x_test, para_data_y_train, para_data_y_test = train_test_split(para_data_x, para_data_y, test_size=0.2, random_state=42)

scaler = StandardScaler()

para_data_x_train_scaled = scaler.fit_transform(para_data_x_train)
para_data_y_train_scaled = scaler.fit_transform(para_data_y_train)

para_data_x_test_scaled = scaler.transform(para_data_x_test)
para_data_y_test_scaled = scaler.transform(para_data_y_test)

from tensorflow.keras.callbacks import TensorBoard

log_dir = "logs/"

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = combined_model.fit(
    [para_data_x_train_scaled, para_data_y_train_scaled],
    [tf.zeros((tf.shape(para_data_x_train_scaled)[0], 10000)), tf.zeros_like(model_psi(para_data_x_train_scaled))],
    validation_data=(
        [para_data_x_test_scaled, para_data_y_test_scaled],
        [tf.zeros((tf.shape(para_data_x_test_scaled)[0], 10000)), tf.zeros_like(model_psi(para_data_x_test_scaled))]
    ),
    verbose=1,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[tensorboard_callback]
)

model_path = 'model.h5'
combined_model.save_weights(model_path)



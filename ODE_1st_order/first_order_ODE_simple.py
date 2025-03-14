# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Main libraries
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""""""

#Defining the PINN
class ODE_1st(tf.keras.Model):

    
    def train_step(self, data):
        
        # x->Training points, y_exact->Analytical (exact) solution at these points
        x,y_exact = data
        
        #Initial conditions for the PINN
        x0 = tf.constant([0.0], dtype=tf.float32)
        y0_exact = tf.constant([1.0], dtype=tf.float32)
        
        #Calculate the gradients and update weights and bias
        with tf.GradientTape() as tape:
            
            #Calculate the gradients dy/dx
            with tf.GradientTape() as tape2:
                tape2.watch(x0)
                tape2.watch(x)
                y0_NN = self(x0, training=True)
                y_NN = self(x, training=True)
            dy_dx_NN = tape2.gradient(y_NN,x)
            
            #Loss = ODE + boundary/initial conditions
            L_d = self.compiled_loss(dy_dx_NN, -y_NN)
            L_b = self.compiled_loss(y0_NN, y0_exact)
            loss = L_d + L_b
                
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(y_exact, y_NN)
        print("ODE Loss:", self.compiled_loss(dy_dx_NN, -y_NN))
        print("Initial Condition Loss:", self.compiled_loss(y0_NN, y0_exact))
        # df_dict={"ode_loss":self.compiled_loss(dy_dx_NN, -y_NN), "IC_loss":self.compiled_loss(y0_NN, y0_exact)}
        # df=pd.DataFrame(df_dict)
        return {m.name: m.result() for m in self.metrics}
    
    
    
#Running the PINN
n_train = 20
xmin = 0
xmax = 4


#Definition of the function domain
x_train = np.linspace(xmin, xmax, n_train)

#The real solution y(x) for training evaluation
y_train = tf.exp(-x_train) 

#Input and output neurons (from the data)
input_neurons = 1
output_neurons = 1
            
#Hyperparameters
epochs = 40

#Definition of the model
activation = 'elu'
input = Input(shape = (input_neurons,))
x = Dense(50, activation = activation)(input)
x = Dense(50, activation = activation)(x)
x = Dense(50, activation = activation)(x)
output = Dense(output_neurons, activation = None)(x)
model = ODE_1st(input, output)

#Definition of the metrics, optimizer and loss
loss = tf.keras.losses.MeanSquaredError()
metrics = tf.keras.metrics.MeanSquaredError()
optimizer = Adam(learning_rate = 0.001)

model.compile(loss = loss, optimizer = optimizer, metrics = [metrics])
model.summary()

#Default
history = model.fit(x_train, y_train, batch_size = 1, epochs = epochs)

# x = np.linspace(1, 4, 1)#Format to try out small values for individual testing
# print(model.predict(x)) #Is giving [[0.3647178]]

# MY own personal addendum
plt.rcParams['figure.dpi'] = 150 #needs to be placed before any axis is plotted
fig1, axs1 = plt.subplots()

# summarize history for loss and metrics
#plt.rcParams['figure.dpi'] = 150
axs1.plot(history.history['loss'], color = 'magenta', label = 'Total losses ($L_D + L_B$')
axs1.plot(history.history['mean_squared_error'], color = 'cyan', label = 'MSE')
axs1.set_ylabel("log")
axs1.set_xlabel('epochs')
axs1.legend(loc = 'upper right')
#axs1.set_title("All three")
plt.show()


# Checking the PINN at different points not included in the training set
n = 500
x = np.linspace(0, 4, n)
y_exact = tf.exp(-x)
y_NN = model.predict(x)

# The gradients (y'(x) and y''(x)) from the model
x_tf = tf.convert_to_tensor(x, dtype = tf.float32)

with tf.GradientTape(persistent  = True) as t:
    t.watch(x_tf)
    with tf.GradientTape(persistent = True) as t2:
        t2.watch(x_tf)
        y = model(x_tf)
    dy_dx_NN = t2.gradient(y, x_tf)
d2y_dx2_NN = t.gradient(dy_dx_NN, x_tf)

# Plot the results
#plt.rcParams['figure.dpi'] = 150
fig2, axs2 = plt.subplots()

axs2.plot(x, y_exact, color = "black", linestyle = 'solid', linewidth = 2.5, 
            label = '$y(x)$ analytical')
axs2.plot(x, y_NN, color = "red", linestyle = 'dashed', linewidth = 2.5,
            label = "$y_{NN}(x)$")
axs2.plot(x, dy_dx_NN, color = "blue", linestyle = '-.', linewidth = 3.0,
            label = "$y'_{NN}(x)$")
axs2.plot(x, d2y_dx2_NN, color = "green", linestyle = 'dotted', linewidth =3.0,
            label = "$y''_{NN}(x)$")
axs2.legend()
axs2.set_xlabel("x")
plt.show()
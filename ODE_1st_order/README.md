1.The ordinary or partial differential equation we need to solve along with the boundary/Initial conditions are incorporated into the loss function of the neural network.

Program: firt_order_ODE_simple.py
This for the simple y' + y(x) = 0 for $$y(x) = e^{-x}$$ to explain the basics of custom neural networking models are being implemented using tensorflow.keras in the program.
Goal is to be able to scale this program to any O.D.E.

Line 19: class ODE_1st(tf.keras.Model):\
Here the class ODE_1st we are defining is employing the inheritance property where it is inheriting the abilities of the pre-defined tf.keras.Model class through which you can create your own custom neural network (can decide how many layers you want, managing weights (line:48), compiling the model (will explain below), training the model (using fit() in line:95), predicting values using the neural network(line:119) and much more that is not currently used in this simple program).

Lines 28-29: (tf.constant's)\


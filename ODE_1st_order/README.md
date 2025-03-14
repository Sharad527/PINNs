1.The ordinary or partial differential equation we need to solve along with the boundary/Initial conditions are incorporated into the loss function of the neural network.

Program: firt_order_ODE_simple.py
This for the simple y' + y(x) = 0 for $$y(x) = e^{-x}$$ to explain the basics of custom neural networking models are being implemented using tensorflow.keras in the program.
Goal is to be able to scale this program to any O.D.E.

Line 19: class ODE_1st(tf.keras.Model):\
1.Here the class ODE_1st we are defining is employing the inheritance property where it is inheriting the abilities of the pre-defined tf.keras.Model class through which you can create your own custom neural network\  2.You can decide how many layers you want, managing weights (line:48), compiling the model (will explain below), training the model (using fit() in line:95), predicting values using the neural network(line:119) and much more that is not currently being used in this simple program.

Lines 28-29: (tf.constant's)\
1.Initial condition (0,1) that the Neural network (apart from the O.D.E.) needs to learn is being through an immutable tensor constant, the shape of this constant is that of a rank-1 tensor (a vector) so we are actually creating 2 1-D arrays [0] and [1] which will be supplied into model (format inputs need to be in as x is one feature so just a 1-D array else there would have been more dimentions).\
2.The purpose a special tf.constant type is being used here because tensorflow wants the constants that should remain unchanged through the entire program should be immutable (meaning the variable value should not be changeable throughout the program, integer arrays are mutable and thus can be modified which is not suitable for tensorflow workflow)

lines

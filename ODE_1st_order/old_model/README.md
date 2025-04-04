1.The ordinary or partial differential equation we need to solve along with the boundary/Initial conditions are incorporated into the loss function of the neural network.

Program: firt_order_ODE_simple.py
This for the simple y' + y(x) = 0 for $$y(x) = e^{-x}$$ to explain the basics of custom neural networking models are being implemented using tensorflow.keras in the program.
Goal is to be able to scale this program to any O.D.E.

Line 19: class ODE_1st(tf.keras.Model):\
1.Here the class ODE_1st we are defining is employing the inheritance property where it is inheriting the abilities of the pre-defined tf.keras.Model class through which you can create your own custom neural network\  2.You can decide how many layers you want, managing weights (line:48), compiling the model (will explain below), training the model (using fit() in line:95), predicting values using the neural network(line:119) and much more that is not currently being used in this simple program.

Lines 28-29: (tf.constant's)\
1.Initial condition (0,1) that the Neural network (apart from the O.D.E.) needs to learn is being through an immutable tensor constant, the shape of this constant is that of a rank-1 tensor (a vector) so we are actually creating 2 1-D arrays [0] and [1] which will be supplied into model (format inputs need to be in as x is one feature so just a 1-D array else there would have been more dimentions).\
2.The purpose a special tf.constant type is being used here because tensorflow wants the constants that should remain unchanged through the entire program should be immutable (meaning the variable value should not be changeable throughout the program, integer arrays are mutable and thus can be modified which is not suitable for tensorflow workflow)

lines 32-45: (tf.GradientTape)\
1.tf.GradientTape() is used to keep track of gradients of the particular variables/parameters that are operating within its block (the code block within the with statement), These gradients are calculated using the weights of the neural nodes and biases to optimize that particular parameter of the network better on the next run.\
2.There are two GradientTape's being used in the program. The first outer one being used as "tape" is used to keep track of the constraints we add onto the neural network for it to learn to solve the differential equation.\
That is why the loss terms (into which one incorporates the constraints one wants for the system) are placed here to be calculated (but they have to come only after the weights and biases of the main neural network have been calculated).\
3.That is why in the inner GradientTape termed as "tape2" tbe main training data is first passed to the self instance\
(it just means we are using the default way of referring to the same class we are working on, very useful if our class has inherited properties like here where our class has inherited the pre-built keras.Model module).\
4.The "tape2" section is the classic neural network structure that is fudamental to build the NN itself, tf.watch(x) is used to keep track of the features being passed into the network.\
This is to make sure the network continues to converge towards the expected output parameter.\
Keeping track of x0 which is the initial value condition is to make sure the NN not only learns the contraint of the differential equation but also what boundary condition it is always supposed to maintain.\
5.Thus the 2 constraints we placed in our formulated problem will be learnt through tape2 while the loss between the values calculated by NN and the actual values we have are collected by tape.\
(compiled_loss is the function which finds the error/loss between predicted and expected values which we can customise in the model.compile(loss = function,...) code in line 91).\
6.The loss is broken up into 2 componenets here because there are 2 constraints the NN is trying to learn, the O.D.E. and the boundary condition we want followed.\
All changes to our code will be here in the tape2.watch() section where we add addition constrainst or boundary conditions we might need,\
the training of the network on the requisite feature space through the self instance,\
the gradient between the feature variables and output parameter (for P.D.E. we will be adding multiple terms here, for now need only 1 line due to it being an O.D.E.)\

lines 47-54:\
1.The network gets updated and is trained to better approach the output parameter by comparing it to expected value here (the gradients we calculated before will provide the needed corrections to the weights of the nodes thus updating the entire state of the NN)\
2.Very little to modify here as it is standard code.

line 91:\
1.The loss function we want while calculating the gradients needed to adjust the weights is supplied here along with the optimizer used. Loss is just Mean squared Error so compiled_loss in tape will come out to as sum of $$|y(0)_NN-y(0)|^2$$ and $$|dy/dx(NN)-(-y(x))|^2$$ as we want to make sure the y' NN is learning comes out to be -y(x) meaning their sum is supposed to minimise to approach this equality.\
2.So as can be seen the major adjustments to code is to be made around how many boundary condition we want to define along with the number of differntial equation constraints the NN is to be penalised through and learn (can increase number of epochs to learn over more cycles but running too many times will lead to overfitting or faulty results)\

For PINN_RLNN_mark1.py the O.D.E. is y'+y = 0 for 0<x<4 and y(0) = 1  

For PINN_RLNN_mark2.py the O.D.E. is y" + y = 0 for 0<x<8, y(0) = 1 and y'(0) = 1  
The loss was reduced for this higher dimension ODE by:  
Increasing learning rate from 3e-2 to 1e-3  
Increasing number of neurons per layer from 32 to 40  
Number of hidden layers from 3 to 4  
Increasing number of collocation points (points to train pde on) from 100 to 1000  




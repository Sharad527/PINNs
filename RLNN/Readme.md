For PINN_RLNN_mark1.py the O.D.E. is y'+y = 0 for 0<x<4 and y(0) = 1  

For PINN_RLNN_mark2.py the O.D.E. is y" + y = 0 for 0<x<8, y(0) = 1 and y'(0) = 1  
The loss was reduced for this higher dimension ODE by:  
Increasing learning rate from 3e-2 to 1e-3  
Increasing number of neurons per layer from 32 to 40  
Number of hidden layers from 3 to 4  
Increasing number of collocation points (points to train pde on) from 100 to 1000  
 
**Very important**  
When oberving the evolution of PINN model in graphs it was noticed it was unable to learn about the second peak, so on increasing weightage of pde_loss and first derivative initial condition loss it was able to start learning to model the curvature more accurately  
This could point to how weightage needs to be distributed in the future when dealing with higher order differential equations  



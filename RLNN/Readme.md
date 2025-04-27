1.For PINN_RLNN_mark1.py the O.D.E. is y'+y = 0 for 0<x<4 and y(0) = 1  

2.For PINN_RLNN_mark2.py the O.D.E. is y" + y = 0 for 0<x<8, y(0) = 1 and y'(0) = 1  
The loss was reduced for this higher dimension ODE by:  
Increasing learning rate from 3e-2 to 0.40e-3  
Increasing number of neurons per layer from 32 to 40  
Number of hidden layers from 3 to 4  
Increasing number of collocation points (points to train pde on) from 100 to 750  
Number of epochs increased to 6000  
Split the weightage given to loss for each Initial condition as this has imporved learning
 
**Note**  
The initial condition y(0) = 1 is being forgotten as network is learning about behavior at upper bound of domain so increased weightage to error from y(0) most so that network will be forced to remember its property as it progresses through 1000s of epochs.  
y'(0) = 1 is given least weightage as it is introducing extreme phase shift into the predictions
pde_loss is given generic loss weightage of 1

<p>
 <img width="520" alt="RLNN_mark2_1,10,0 001andlr=0 4e-3" src="https://github.com/user-attachments/assets/694ea93e-5ac6-4c97-a491-c202cd3b78fa" />
</p>

3.Adv_RLNN.py can solve the laguerre equation with a reasonable degree of accuracy for n=2 to n=6, haven't tested n>6 yet

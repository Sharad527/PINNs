The neural network was purposely built from scratch like this as it gives a better idea of what is going on and gives you the ability to control down to the number of connections between each layer of the NN
Highy customisable too and can test and implement ideas gathered from a wide variety of research papers.

NN_mark{i} files are benchmarks used to compare with PINN_mark{i} files and check what modifications to loss2 function can better model the P.D.E. solver

This model has 2 losses in the PINN programs, the first one being the error between y-value predicted by the NN and the y-value from training dataset, this is a simple Mean Square Error stored as loss1.
The second loss is the residual loss from physics constraints (P.D.E. and B.C.s) stored in loss2.

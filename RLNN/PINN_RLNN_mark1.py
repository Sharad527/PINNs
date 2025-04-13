#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 10:52:16 2025

@author: aravind

RLNN means Residual Loss Network (My Shorthand for No training data PINNS)

Working on a simple O.D.E. dy/dx + y = 0 (0<x<4) for y(0) = 1

"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Just for reference
def rk4():
    nsteps = 3000 # Number of time steps
    xMin, xMax = 0.0,4.0 # x domain
    xStep = (xMax-xMin)/nsteps # x step
    xPoints = np.arange(xMin, xMax, xStep)
    
    #Setting the initial conditions
    y0 = 1.0
    y = [y0]
    
    #Solving through x
    
    for x in xPoints[1:]:
        k1 = -y[-1]
        k2 = -(y[-1] + 0.5*xStep*k1)
        k3 = -(y[-1] + 0.5*xStep*k2)
        k4 = -(y[-1] + xStep*k3)
        
        y.append(y[-1] + (xStep/6)*(k1 + 2*k2 + 2*k3 + k4))
    
    return xPoints,y
        
x,y = rk4()

def plot_result(x,y,x_c,y_pred):
    
    plt.figure(figsize = (8,4))

    plt.scatter(x_c, y_pred, color="red", linewidth=2, alpha=0.8, label="PINN prediction")
    plt.plot(x,y, color="blue", linewidth=2, alpha=0.8, linestyle='--', label="Exact Solution")
    # l=plt.legend(loc=(0.67,0.62), frameon=False, fontsize="large")
    # plt.setp(l.get_texts(), color="k")
    # plt.xlim(-1.25,4.5)
    # plt.ylim(-0.65,1.0)
    plt.text(2.965,1.95,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
    plt.ylabel('y',fontsize="xx-large")
    plt.xlabel('x',fontsize="xx-large")
    plt.axis("on")
    plt.legend()


class FCN(nn.Module): #Forward Connected Network
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.ELU()
        
        self.fcs = nn.Sequential(*[nn.Linear(N_INPUT, N_HIDDEN), activation])
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN), activation]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    
#Coefficient of y(x) in O.D.E.
lam = 1 

# Collocation points are needed to train network on Physics constraints
xPoints = np.random.rand(50)*4
x_c = torch.Tensor(xPoints).view(-1,1).requires_grad_(True)
# x_c = torch.linspace(0,4,300).view(-1,1).requires_grad_(True)

# Initial Conditions
I_c = np.array([[0,1]])
x_ic = torch.Tensor([0]).view(-1,1)
y_true_ic = torch.Tensor([1]).view(-1,1)

# Any other boundary conditions you want to implement
# Supply them through format mentioned above
# B_c = [[...,...],...]
# x_bc = torch.Tensor(B_c[:,0]).view(-1,1)
# y_bc = torch.Tensor(B_c[:,1]).view(-1,1)

torch.manual_seed(123)

model = FCN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr = 3e-2)
# lr is learning rate

files = []
loss_history = {"pde_loss":[],"ic_loss":[],"bc_loss":[],"total_loss":[]}


# Weights for the loss terms when calculating total loss
w_pde, w_ic = 1.0, 0.1
W_bc = 1.0

for i in range(100):
    
    optimizer.zero_grad()
    
    # Computing Physics loss through P.D.E. on collocation points
    y_pred_c = model(x_c)
    dy_dx = torch.autograd.grad(y_pred_c, x_c, grad_outputs=torch.ones_like(y_pred_c), create_graph = True)[0]
    
    pde = (dy_dx +lam*y_pred_c) # dy/dx + y(x) = 0
    pde_loss = torch.mean(pde**2)
    
    # Same flow for boundary conditions as above
    
    # Computing Physics loss through Initial Conditions
    y_pred_ic = model(x_ic)
    
    IC_loss = torch.mean((y_pred_ic - y_true_ic)**2)
    
    # Adding the total loss here
    total_loss = w_pde*pde_loss + w_ic*IC_loss #+ w_bs*BC_loss
    
    total_loss.backward()
    optimizer.step()
    
    if (i+1) % 10 == 0:
        loss_history["pde_loss"].append(pde_loss.detach())
        loss_history["ic_loss"].append(IC_loss.detach())
        # loss_history["bc_loss"].append(BC_loss.detach())
        loss_history["total_loss"].append(total_loss.detach())
        
        xCheck = np.random.rand(300)*4
        x_check = torch.Tensor(xCheck).view(-1,1).requires_grad_(True)
        
        y_pred = model(x_check)
        
        x_axis = x_check.detach().numpy()
        y_axis = y_pred.detach().numpy()
            
        plot_result(x,y,x_axis,y_axis)
                
        # if (i+1) % 10 == 0: plt.show()
        # else: plt.close("all")

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(loss_history["pde_loss"], label="PDE loss")

ax.plot(loss_history["ic_loss"], label="IC loss")

#ax.plot(loss_history["bc_loss"], label="BC loss")

ax.plot(loss_history["total_loss"], label="Total loss")
ax.set_yscale('log')

ax.set_xlabel('Training step ($10^2$)',fontsize="xx-large")
ax.set_ylabel('Loss terms',fontsize="xx-large")
ax.legend()
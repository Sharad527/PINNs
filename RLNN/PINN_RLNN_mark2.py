#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 16:20:32 2025

@author: aravind

RLNN means Residual Loss Network (My Shorthand for No training data PINNS)

Working on a simple O.D.E. y" + y = 0 (0<=x<=8) for y(0) = 1, y'(0)=1

"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Just for reference
def rk4():
    nsteps = 3000 # Number of time steps
    xMin, xMax = 0.0,8.0 # x domain
    #np.linspace() includes lower and upper limit specified
    xStep = (xMax-xMin)/nsteps # x step
    xPoints = np.linspace(xMin, xMax, nsteps + 1)  # +1 to include xMax
    
    # Initial conditions: y(0) = 1.0, y'(0) = 0.0 (starting at rest)
    y0 = 1.0
    v0 = 0.0  # y'(0) = v(0)
    
    y = [y0]  # Stores y(x)
    v = [v0]  # Stores y'(x) = v(x)
    
    for i in range(nsteps):
        # Current values
        y_curr = y[-1]
        v_curr = v[-1]
        
        # RK4 for y' = v (k1, k2, k3, k4 for y)
        k1_y = v_curr
        k1_v = -y_curr
        
        k2_y = v_curr + 0.5 * xStep * k1_v
        k2_v = -(y_curr + 0.5 * xStep * k1_y)
        
        k3_y = v_curr + 0.5 * xStep * k2_v
        k3_v = -(y_curr + 0.5 * xStep * k2_y)
        
        k4_y = v_curr + xStep * k3_v
        k4_v = -(y_curr + xStep * k3_y)
        
        # Update y and v
        y_new = y_curr + (xStep / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        v_new = v_curr + (xStep / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        
        y.append(y_new)
        v.append(v_new)
    
    return xPoints,y
        
x,y = rk4()

def plot_result(x,y,x_c,y_pred):
    
    plt.figure(figsize = (8,4))

    plt.scatter(x_axis, y_axis, color="red", linewidth=0.2, alpha=0.8, label="PINN prediction")
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
    
# Importance of requires_grad_(True)
# PyTorch needs to track gradients for the input tensor x_ic when computing derivatives 
# using torch.autograd.grad() so specifying to pytorch x_ic or x_c is used for derivatives
# This must be like tf.watch(x_ic) in tensorflow where the variables whose gradients are taken for
# backpropagation needs to be mentioned to the network

# Collocation points are needed to train network on Physics constraints
xPoints = np.random.rand(500)*8
x_c = torch.Tensor(xPoints).view(-1,1).requires_grad_(True)

# Initial Conditions
x_ic = torch.Tensor([0]).view(-1,1).requires_grad_(True) #x = 0 for below two conditions
y_true_ic = torch.Tensor([1]).view(-1,1) #y(0) =1
dy_dx_true_ic = torch.Tensor([1]).view(-1,1) #dy(0)/dx = 1

# Any other boundary conditions you want to implement
# Supply them through format mentioned above
# B_c = [[...,...],...]
# x_bc = torch.Tensor(B_c[:,0]).view(-1,1)
# y_bc = torch.Tensor(B_c[:,1]).view(-1,1)

torch.manual_seed(123)

model = FCN(1,1,40,4)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.40e-3)
# lr is learning rate

files = []
loss_history = {"pde_loss":[],"ic_1_loss":[],"ic_2_loss":[],"bc_loss":[],"total_loss":[]}

# Weights for the loss terms when calculating total loss
w_pde = 1.0#1.5
w_ic_1 = 10.0#1.5#0.1 # For each initial condition added a weight to adjust later on
w_ic_2 = 0.001#0.01#0.4
W_bc = 1.0

#Coefficient of y(x) in O.D.E.
lam = 1 

for i in range(6000):
    
    optimizer.zero_grad()
    
    # Computing Physics loss through P.D.E. on collocation points
    y_pred_c = model(x_c)
    dy_dx = torch.autograd.grad(y_pred_c, x_c, grad_outputs=torch.ones_like(y_pred_c), create_graph = True)[0]
    d2y_dx2 = torch.autograd.grad(dy_dx, x_c, grad_outputs=torch.ones_like(dy_dx), create_graph = True)[0]
    
    pde = (d2y_dx2 +lam*y_pred_c) # y" + y(x) = 0
    pde_loss = torch.mean(pde**2)
    
    # Same flow for boundary conditions as below
    
    # Computing Physics loss through Initial Conditions
    # y(0) = 1
    y_pred_ic = model(x_ic)
    
    # y'(0) = 1 (using same x_ic and y_pred_ic as we want derivative at same point to give 1)
    # Remember you get a tuple on auto-differentiation so take value at index 0
    dy_dx_pred_ic = torch.autograd.grad(y_pred_ic, x_ic, grad_outputs=torch.ones_like(y_pred_ic), create_graph= True)[0]
    
    #Summing losses of all ICs (can do them seperately too with individual weights if this method isn't reducing IC_loss satifactorily)
    IC_1_loss = w_ic_1*torch.mean((y_pred_ic - y_true_ic)**2)
    IC_2_loss = w_ic_2*torch.mean((dy_dx_pred_ic - dy_dx_true_ic)**2)
    IC_loss = IC_1_loss + IC_2_loss
    
    # Adding the total loss here
    total_loss = w_pde*pde_loss + IC_loss #+ w_bs*BC_loss
    
    total_loss.backward()
    optimizer.step()
    
    if (i+1) % 20 == 0:
        loss_history["pde_loss"].append(pde_loss.detach())
        loss_history["ic_1_loss"].append(IC_1_loss.detach())
        loss_history["ic_2_loss"].append(IC_2_loss.detach())
        # loss_history["bc_loss"].append(BC_loss.detach())
        loss_history["total_loss"].append(total_loss.detach())
        
        xCheck = np.random.rand(750)*8
        x_check = torch.Tensor(xCheck).view(-1,1)
        
        y_pred = model(x_check)
        
        x_axis = x_check.detach().numpy()
        y_axis = y_pred.detach().numpy()
            
        plot_result(x,y,x_axis,y_axis)
                
        # if (i+1) % 10 == 0: plt.show()
        # else: plt.close("all")

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(loss_history["pde_loss"], label="PDE loss")

ax.plot(loss_history["ic_1_loss"], label="IC_1 loss")
ax.plot(loss_history["ic_2_loss"], label="IC_2 loss")

#ax.plot(loss_history["bc_loss"], label="BC loss")

ax.plot(loss_history["total_loss"], label="Total loss")
ax.set_yscale('log')

ax.set_xlabel('Training step ($10^2$)',fontsize="xx-large")
ax.set_ylabel('Loss terms',fontsize="xx-large")
ax.legend()


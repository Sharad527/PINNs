#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 02:37:34 2025

@author: aravind

Working on a simple O.D.E. y" - y -3y^2= 0 for y(0) = -1/2, y'(0)=0

conda install pytorch torchvision torchaudio -c pytorch
"""

#from PIL import Image #Don't know for what this is

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def rk4():
    nsteps = 3000 # Number of time steps
    xMin, xMax = -5.0,5.0 # x domain
    #np.linspace() includes lower and upper limit specified
    xStep = (xMax-xMin)/nsteps # x step
    xPoints = np.linspace(xMin, xMax, nsteps + 1)  # +1 to include xMax
    
    # Initial conditions: y(0) = y0, y'(0) = v0
    y0 = -0.5  
    v0 = 0.0  

    y = [y0]  # Stores y(x)
    v = [v0]  # Stores y'(x) = v(x)

    for i in range(nsteps):
        y_curr = y[-1]
        v_curr = v[-1]

        # RK4 coefficients for y' = v
        k1_y = v_curr
        k1_v = y_curr + 3 * y_curr**2  # From y'' = y + 3y²

        k2_y = v_curr + 0.5 * xStep * k1_v
        k2_v = (y_curr + 0.5 * xStep * k1_y) + 3 * (y_curr + 0.5 * xStep * k1_y)**2

        k3_y = v_curr + 0.5 * xStep * k2_v
        k3_v = (y_curr + 0.5 * xStep * k2_y) + 3 * (y_curr + 0.5 * xStep * k2_y)**2

        k4_y = v_curr + xStep * k3_v
        k4_v = (y_curr + xStep * k3_y) + 3 * (y_curr + xStep * k3_y)**2

        # Update y and v using weighted average
        y_new = y_curr + (xStep / 6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        v_new = v_curr + (xStep / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        y.append(y_new)
        v.append(v_new)

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(xPoints, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return x_tensor, y_tensor


class FCN(nn.Module): #Forward Connected Network
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh()
        
        self.fcs = nn.Sequential(*[nn.Linear(N_INPUT, N_HIDDEN), activation])
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN), activation]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


x,y = rk4()
x = x[::1]
y = y[::1]

x_data = x[0:3000:80]
y_data = y[0:3000:80]

# plt.figure()
# plt.plot(x, y, label = "Exact Solution y")
# plt.scatter(x_data, y_data, color = "tab:orange", label = "Training Data")
# plt.legend()
# plt.show()


# Constructing the NN frome here

def plot_result(x,y,x_data,y_data,yh,xp=None):
    
    plt.figure(figsize = (8,4))
    plt.plot(x,yh, color="tab:red", linewidth=2, alpha=0.8, label="NN prediction")
    plt.plot(x,y, color="blue", linewidth=2, alpha=0.8, linestyle='--', label="Exact Solution")
    plt.scatter(x_data, y_data, s=60, color="tab:red", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=30, color="tab:green", alpha=0.4, label="Collocation points")
    l=plt.legend(loc=(0.67,0.62), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    # plt.xlim(-1.25,4.5)
    # plt.ylim(-0.65,1.0)
    plt.text(2.965,0.25,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
    plt.ylabel('y',fontsize="xx-large")
    plt.xlabel('Time',fontsize="xx-large")
    plt.axis("on")


# We choose the colocation points for physics ......................................
x_physics = torch.linspace(-5,5,50).view(-1,1).requires_grad_(True)# sample locations over the problem domain
lam=1

torch.manual_seed(123)
model = FCN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=3e-3)
files = []

loss_history = []
loss2_history = []
for i in range(200):
    optimizer.zero_grad()
    
    # We compute the "data loss" .............................................
    yh = model(x_data)
    loss1 = 0.999 * torch.mean((yh - y_data)**2)

    # Physics loss (enforcing y'' - y - 3y² = 0)
    yhp = model(x_physics)  # Predictions at collocation points

    # First derivative dy/dx
    dy_dx = torch.autograd.grad(yhp, x_physics,grad_outputs=torch.ones_like(yhp), 
    create_graph=True)[0]  
    # Required for higher-order derivatives


    # Second derivative d²y/dx²
    d2y_dx2 = torch.autograd.grad(dy_dx, x_physics,grad_outputs=torch.ones_like(dy_dx), 
    create_graph=True)[0]

    # Physics residual: y'' - y - 3y² = 0
    physics_residual = d2y_dx2 - yhp - 3 * yhp**2
    loss2 = 0.001 * torch.mean(physics_residual**2)

    # Total loss (weighted combination)
    loss = loss1 + loss2

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    
    # We plot the result as training progresses ....................................
    if (i+1) % 10 == 0:
        loss_history.append(loss.detach())
        loss2_history.append(loss2.detach())

        yh = model(x).detach()
        xp = x_physics.detach()        
        plot_result(x,y,x_data,y_data,yh,xp)
                
        if (i+1) % 10 == 0: plt.show()
        else: plt.close("all")
            
plt.plot(loss_history)
plt.xlabel('Training step ($10^2$)',fontsize="xx-large")
plt.ylabel('Loss',fontsize="xx-large")
plt.yscale('log')
plt.legend()

plt.plot(loss2_history, label="loss2")
plt.yscale('log')
plt.legend()
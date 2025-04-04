#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 02:12:34 2025

@author: aravind

Working on a simple O.D.E. y" + y = 0 (0<=x<=8) for y(0) = 1, y'(0)=1

conda install pytorch torchvision torchaudio -c pytorch
"""

#from PIL import Image #Don't know for what this is

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
    
    # Convert to PyTorch tensors
    x_tensor = torch.Tensor(xPoints).view(-1, 1)
    y_tensor = torch.Tensor(y).view(-1, 1)
    
    return x_tensor, y_tensor


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
    plt.text(2.965,1.95,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
    plt.ylabel('y',fontsize="xx-large")
    plt.xlabel('Time',fontsize="xx-large")
    plt.axis("on")
    

# train standard neural network to fit training data ..................
torch.manual_seed(123)
model = FCN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=3e-3)
files = []
loss11_history = []

for i in range(200):
    optimizer.zero_grad()
    yh = model(x_data)
  
    loss = torch.mean((yh-y_data)**2)  # use mean squared error
    loss.backward()
    optimizer.step()
    
    
    # plot the result as training progresses ......................
    if (i+1) % 10 == 0: 
        loss11_history.append(loss.detach())  
        yh = model(x).detach()
              
        plot_result(x,y,x_data,y_data,yh)
        
    
        if (i+1) % 10 == 0: 
            plt.show()
        else: 
            plt.close("all")
            
fig11 = plt.figure(11)
plt.plot(loss11_history)
plt.xlabel('Training step ($10^2$)',fontsize="xx-large")
plt.ylabel('Loss',fontsize="xx-large")
plt.yscale('log')
plt.legend()

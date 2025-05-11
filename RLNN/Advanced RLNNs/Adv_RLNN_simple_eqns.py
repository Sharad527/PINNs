#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:35:26 2025

@author: aravind 

Testing for simpler equations like:-
1) y" + y - x^2 - 2 = 0 -> y = x^2 + 10 * sin(x)
  
2) y" + y + x = 0
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os # For creating directories if needed

# Importing initialization modules for weight initiliaization
import torch.nn.init as init

# plotting laguerre onto prediction with grid
from scipy.special import eval_laguerre

torch.manual_seed(42)
np.random.seed(42)

# Detecting device, GPU works 2x faster at the minimum
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    # Set seed for CUDA specifically if using GPU
    torch.cuda.manual_seed_all(42)
    
# --- Plotting Function ---
def plot_result(x_exact, y_exact, x_scatter, y_scatter, current_epoch, poly_n):
    """Plots the exact solution vs PINN prediction points."""
    plt.figure(figsize=(10, 6)) # Adjusted size

    plt.scatter(x_scatter, y_scatter, color="red", s=10, alpha=0.6, label="PINN Prediction Points")
    plt.plot(x_exact, y_exact, color="blue", linewidth=2.5, alpha=0.8, linestyle='--', label=f"Exact $L_{poly_n}(x)$")

    # Change this if not dealing with Laguerre polynomial
    plt.title(f"Laguerre Polynomial n={poly_n} Prediction vs Exact")
    plt.text(0.97, 0.97, f"Epoch: {current_epoch+1}", fontsize="medium", color="k",
             ha='right', va='top', transform=plt.gca().transAxes) # Position relative to axes
    plt.ylabel('$y(x)$', fontsize="large")
    plt.xlabel('$x$', fontsize="large")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize="medium")

    # Dynamic Y limits based on exact solution range + margin
    min_y = np.min(y_exact)
    max_y = np.max(y_exact)
    margin = (max_y - min_y) * 0.15 + 0.5 # Adjusted margin
    plt.ylim(min_y - margin, max_y + margin)
    plt.tight_layout() # Adjust layout
    plt.show()
    

# Main Neural Network
class FCN(nn.Module):
    """Fully Connected Network for the PINN."""
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, initializer='xavier'):
        super().__init__()
        if N_LAYERS < 1:
            raise ValueError("N_LAYERS must be at least 1")

        self.initializer = initializer
        activation = nn.Tanh() # Tanh often works well for PINNs

        layers = []
        layers.append(nn.Linear(N_INPUT, N_HIDDEN))
        layers.append(activation)
        for _ in range(N_LAYERS - 1):
            layers.append(nn.Linear(N_HIDDEN, N_HIDDEN))
            layers.append(activation)
        layers.append(nn.Linear(N_HIDDEN, N_OUTPUT))

        self.net = nn.Sequential(*layers)
        self.init_weights() # Call initialization

    def init_weights(self):
        """Initializes weights based on the chosen initializer type."""
        with torch.no_grad():
            print(f"Initializing weights using {self.initializer} uniform...")
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    if self.initializer == 'xavier':
                        # Xavier is used by default as it works best with Tanh
                        init.xavier_uniform_(m.weight)
                    elif self.initializer == 'kaiming':
                        # Kaiming is often used with ReLU but can be tried for Tanh
                        init.kaiming_uniform_(m.weight, nonlinearity='tanh')
                    # Add other initializers like 'orthogonal' or 'normal' here if needed
                    else:
                        print(f"Warning: Unknown initializer '{self.initializer}'. Using default.")
                        # Default initialization will be used if none specified
                        pass # Let PyTorch use its default Linear layer init

                    if m.bias is not None:
                        init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# Computing losses seperately to find changes quicker
def calculate_losses(model, x_c, x_bdy, n):
    """Calculates ODE residual loss and boundary condition losses."""
    # Collocation points loss (ODE Residual)
    x_c.requires_grad_(True)
    y_pred_c = model(x_c)
    dy_dx_c = torch.autograd.grad(y_pred_c, x_c, grad_outputs=torch.ones_like(y_pred_c), create_graph=True)[0]
    dy2_dx2_c = torch.autograd.grad(dy_dx_c, x_c, grad_outputs=torch.ones_like(dy_dx_c), create_graph=True)[0]

    # Ensure x_c > 0 for the first term or handle x=0 case carefully if included in x_c
    # Below
    #ode_residual = x_c * dy2_dx2_c + (1.0 - x_c) * dy_dx_c + n * y_pred_c
    ode_residual = dy2_dx2_c + y_pred_c + x_c #dy2_dx2_c + y_pred_c - x_c**2 - 2
    ode_loss = torch.mean(ode_residual**2)

    # Boundary points loss (Initial Conditions at x=0)
    x_bdy.requires_grad_(True)
    y_pred_bdy = model(x_bdy)

    # Loss for y(0) = 1
    tru_y_bdy_val = 10 * torch.ones_like(y_pred_bdy)
    bc_loss = torch.mean((y_pred_bdy - tru_y_bdy_val)**2) # torch.tensor([[0]], dtype=torch.float32, device=device) 

    # Loss for y'(0) = -n
    dy_dx_bdy = torch.autograd.grad(y_pred_bdy, x_bdy, grad_outputs=torch.ones_like(y_pred_bdy), create_graph=True)[0]
    tru_y_deriv_bdy_val = -1 * torch.ones_like(y_pred_bdy) # -float(n) * torch.ones_like(dy_dx_bdy)
    deriv_bc_loss = torch.mean((dy_dx_bdy - tru_y_deriv_bdy_val)**2) # torch.tensor([[0]], dtype=torch.float32, device=device)

    # Detach inputs after use (reduces memory slightly)
    x_c.requires_grad_(False)
    x_bdy.requires_grad_(False)

    return ode_loss, bc_loss, deriv_bc_loss



# Using main method in programs from now as it allows the program to be called as a module and can
# be used as a package for larger future projects
if __name__ == "__main__":

    # --- Hyperparameters ---
    N_INPUT = 1       # Input 
    N_OUTPUT = 1      # Output
    N_HIDDEN = 64     # Neurons per hidden layer
    N_LAYERS = 4      # Number of hidden layers

    n = 4             # Degree of the Laguerre polynomial
                      # If using modified version of this program for simpler eqn.s
                      # set n=4 to get maximum minimization of loss

    # Try 'kaiming' if 'xavier' struggles for higher n
    initializer_choice = 'xavier'

    adapt_weight_enabled = True   # Set to False to use fixed initial weights

    w_ode = np.power(10,n-1)#1000.0        # Reference weight (can be non-1.0)
    w_ic_1 = np.power(10,n+1)#50000.0       # Initial weight for y(0)=1 loss
    w_ic_2 = np.power(10,n)#50000.0 # Initial weight for y'(0)=-n loss

    # Smaller LR or different scheduler params might be needed for higher n
    
    ini_lr = 1.25e-3
    learning_rate = ini_lr          # Initial learning rate
    fin_lr = 1.0e-4
    lr_gamma = 0.975                 # LR decay factor
    epochs = 25000                  # Total training epochs
    lr_step_size = np.floor(epochs/np.emath.logn(lr_gamma,fin_lr/ini_lr))    

    # --- Domain and Data Generation ---
    x_min = 0.0
    x_max = 8.0                     # Domain for x: [x_min, x_max]
    num_collocation_points = 500   # Number of points for ODE loss calculation

    # Boundary condition point x=0 (on the correct device)
    x_bdy = torch.tensor([[0]], dtype=torch.float32, device=device)

    # Collocation points shifted into epochs loop to see if they can learn information well still

    # Model constructedm optimizer initialized and scheduler fine tuned
    model = FCN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, initializer=initializer_choice).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    print("\n--- Training Configuration ---")
    print(f"Network: {N_LAYERS} hidden layers, {N_HIDDEN} neurons each")
    print(f"Initializer: {initializer_choice}")
    print(f"Domain: [{x_min}, {x_max}], Collocation Points: {num_collocation_points}")
    print(f"Epochs: {epochs}, Initial LR: {learning_rate}, Scheduler Step: {lr_step_size}, Gamma: {lr_gamma}")
    print("----------------------------\n")

    # --- Training Loop ---
    start_time = time.time()

    # History tracking
    loss_history = {"ode_loss":[], "bc_loss":[], "deriv_bc_loss":[]}
    total_loss_history = []
    weight_history = {"bc": [w_ic_1], "deriv_bc": [w_ic_2]} # Track weights

    # Initialize current weights
    w_ode = w_ode
    w_ic_1 = w_ic_1
    w_ic_2 = w_ic_2

    epoch_check = 1000 # How often to print status and optionally plot

    for epoch in range(epochs):
        
        # Collocation points (randomly sampled, on the correct device)
        # Exclude x=0 slightly if worried about x*y'' term numerically, although mathematically okay here.
        # x_c_np = np.random.uniform(x_min + 1e-6, x_max, num_collocation_points)
        x_c_np = np.random.rand(num_collocation_points) * (x_max - x_min) + x_min
        x_c = torch.tensor(x_c_np, dtype=torch.float32, device=device).view(-1, 1)

        model.train() # Set model to training mode
        optimizer.zero_grad() # Clear gradients

        # Calculate individual losses (unweighted)
        ode_loss, bc_loss, deriv_bc_loss = calculate_losses(model, x_c, x_bdy, n)

        # Store raw losses for adaptive logic and plotting
        loss_history["ode_loss"].append(ode_loss.item())
        loss_history["bc_loss"].append(bc_loss.item())
        loss_history["deriv_bc_loss"].append(deriv_bc_loss.item())


        if epoch > 0: # Avoid appending initial value twice
             weight_history["bc"].append(w_ic_1)
             weight_history["deriv_bc"].append(w_ic_2)

        # Calculate total weighted loss using potentially adapted weights
        total_loss = w_ode * ode_loss + w_ic_1 * bc_loss + w_ic_2 * deriv_bc_loss

        # Backpropagation and Optimizer Step
        total_loss.backward()
        optimizer.step()
        scheduler.step() # Step the learning rate scheduler

        # Store total loss
        total_loss_history.append(total_loss.item())

        # --- Logging and Intermediate Plotting ---
        if (epoch + 1) % epoch_check == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}], LR: {current_lr:.2e}, Loss: {total_loss.item():.4e}, "
                  f"ODE: {ode_loss.item():.3e}, BC: {bc_loss.item():.3e}, BC': {deriv_bc_loss.item():.3e} "
                  f"(λ_BC: {w_ic_1:.1f}, λ_BC': {w_ic_2:.1f})")

            model.eval() # Switch to evaluation mode for prediction
            with torch.no_grad(): # Disable gradient tracking

                num_plot_points = 500
                x_plot_scatter_np = np.random.rand(num_plot_points) * (x_max - x_min) + x_min
                x_plot_scatter = torch.tensor(x_plot_scatter_np, dtype=torch.float32, device=device).view(-1, 1)
                y_pred_scatter = model(x_plot_scatter)

                # Generate points for the exact solution line
                x_exact_plot = np.linspace(x_min, x_max, 300)
                # y_exact_plot = eval_laguerre(n, x_exact_plot)
                y_exact_plot = -x_exact_plot + 10 * np.cos(x_exact_plot)#x_exact_plot**2 + 10 * np.sin(x_exact_plot)

                # Prepare data for plotting (move to CPU, convert to NumPy)
                x_scatter_np = x_plot_scatter.cpu().numpy()
                y_scatter_np = y_pred_scatter.cpu().numpy()

                # Plotting results
                plot_result(x_exact_plot, y_exact_plot, x_scatter_np, y_scatter_np, epoch, n)

            model.train() # Switch back to training mode

    # Training done taking end time to measure time elapse
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"\n--- Training Finished ---")
    print(f"Total time: {time_taken:.2f} seconds.")

# %%
    # Final smooth plot as results
    print("Generating final prediction plot...")
    model.eval()
    with torch.no_grad():
        num_final_points = 500
        x_final_np = np.linspace(x_min, x_max, num_final_points)
        x_final = torch.tensor(x_final_np, dtype=torch.float32, device=device).view(-1, 1)
        y_final_pred = model(x_final)

        #y_exact_final = eval_laguerre(n, x_final_np)
        y_exact_final = -x_final_np + 10 * np.cos(x_final_np)#x_final_np**2 + 10 * np.sin(x_final_np)
        y_final_pred_np = y_final_pred.cpu().numpy()

        # Calculate L2 relative error
        l2_error = np.linalg.norm(y_final_pred_np.flatten() - y_exact_final.flatten()) / np.linalg.norm(y_exact_final.flatten())
        print(f"Final L2 Relative Error: {l2_error:.4e}")



        # Use the same plot function, plotting the final prediction as a line now
        plt.figure(figsize=(10, 6))
        plt.plot(x_final_np, y_exact_final, color="blue", linewidth=2.5, alpha=0.8, linestyle='--', label=f"Exact $L_{n}(x)$")
        plt.plot(x_final_np, y_final_pred_np, color="red", linewidth=2, alpha=0.8, label="PINN Final Prediction")
        plt.title(f"Final Comparison: Laguerre Polynomial n={n} (L2 Error: {l2_error:.2e})")
        plt.ylabel('$y(x)$', fontsize="large")
        plt.xlabel('$x$', fontsize="large")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize="medium")
        min_y = np.min(y_exact_final)
        max_y = np.max(y_exact_final)
        margin = (max_y - min_y) * 0.15 + 0.5
        plt.ylim(min_y - margin, max_y + margin)
        plt.tight_layout()
        plt.show()


    # --- Plotting Loss and Weight History ---
    print("Generating loss and weight history plot...")
    fig, ax1 = plt.subplots(figsize=(12, 7)) # Adjusted size

    epochs_axis = range(1, epochs + 1)

    # Plot Losses on primary y-axis (log scale)
    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize="large")
    ax1.set_ylabel('Loss (Log Scale)', fontsize="large", color=color)
    # Plot raw losses with more transparency
    ax1.plot(epochs_axis, loss_history["ode_loss"], label="ODE Loss (Raw)", alpha=0.4, color='red', linestyle=':')
    ax1.plot(epochs_axis, loss_history["bc_loss"], label="BC Loss (Raw)", alpha=0.4, color='green', linestyle=':')
    ax1.plot(epochs_axis, loss_history["deriv_bc_loss"], label="BC Deriv Loss (Raw)", alpha=0.4, color='blue', linestyle=':')
    # Plot total weighted loss prominently
    ax1.plot(epochs_axis, total_loss_history, label="Total Loss (Weighted)", linewidth=2, color=color)
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color, labelsize='medium')
    ax1.tick_params(axis='x', labelsize='medium')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='center right', fontsize='medium')

    ax1.set_title(f"Training History (n={n}, Time: {time_taken:.0f}s)", fontsize="x-large")
    fig.tight_layout() # Adjust layout to prevent overlap
    plt.show()

    print("\n--- End of Run ---")
# %%
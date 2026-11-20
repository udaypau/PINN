import torch
import torch.nn as nn

import math 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
"""
        p* = (p*h)/(mu0)  where mu0 = refrence values
        y*  = y/h                    h = height of pipe
        x*  = x*10/l                  l =  length of pipe
        10 is provided since that is our aspect ratio
        L = h*10/l
"""
#problem paramets
Pnd_in = 7  #velocity pressure
Pnd_out = 0  #velocity pressure
h = 1        #half width of channel
l  = 10      #length of channel
mu0 = 0.1     #viscosity mu0

a = 10 #aspect ratio
L = h*a/l
#williamson model
k = 1.2 #consistency
n = 0.5 #powerlaw

class PINN_Network(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=3, neurons_per_layer=90):
        super().__init__()
        
        layers = []
        
        # 1. Input Layer
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(nn.Tanh())
        
        # 2. Hidden Layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
            
        # 3. Output Layer (No activation function here to allow unbounded physics predictions)
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        
        # Compile into a PyTorch Sequential model
        self.network = nn.Sequential(*layers)
        
        # Apply Xavier/Glorot initialization (Standard best practice for Tanh PINNs)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, xy):
        """
        x: A tensor of spatial/temporal coordinates (e.g., shape [N, 1] for y-only, [N, 2] for x,y)
        Returns: The predicted physics variables (e.g., shape [N, 1] for just u)
        """
        return self.network(xy)
    @staticmethod
    def grad(f, xy):        # <---------------------------------- Differential operator
        """Gradient of scalar field f (N,1) w.r.t. xy (N,2) -> (N,2): [d/dx*, d/dy*]"""
        return torch.autograd.grad(f, xy, grad_outputs=torch.ones_like(f),
                               create_graph=True)[0]
    
##########################################################################################################
#                                     GEOMETRY SECTION                                                   #
##########################################################################################################
N = 80
ey = 1 / N
ex = a / N  # aspect ratio into consideration

x_vals = torch.linspace(0 + ex, a - ex, N)
y_vals = torch.linspace(-1 + ey, 1 - ey, N)

X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
X0 = torch.full_like(X, 1 / 2)
y0 = torch.full_like(Y, 0)

X_flat = X.flatten().unsqueeze(1)
Y_flat = Y.flatten().unsqueeze(1)
interior_xy = torch.cat([X_flat, Y_flat], dim=1)
interior_xy.requires_grad = True  # FIX 1: required for autograd inside Loss()


# ==========================================
# BOUNDARY GENERATION
# ==========================================
N_b = 200

# 1. Top and Bottom Walls
x_wall = torch.linspace(0, a, N_b).unsqueeze(1)

top_wall_y = torch.full_like(x_wall, 1.0)
top_wall_xy = torch.cat([x_wall, top_wall_y], dim=1)

bottom_wall_y = torch.full_like(x_wall, -1.0)
bottom_wall_xy = torch.cat([x_wall, bottom_wall_y], dim=1)

# No-Slip Walls (u=0, v=0)
walls_xy = torch.cat([top_wall_xy, bottom_wall_xy], dim=0)
walls_xy.requires_grad = True

# 2. Inlet and Outlet
y_flow = torch.linspace(-1, 1, N_b).unsqueeze(1)

inlet_x = torch.full_like(y_flow, 0.0)
inlet_xy = torch.cat([inlet_x, y_flow], dim=1)
inlet_xy.requires_grad = True

outlet_x = torch.full_like(y_flow, a)
outlet_xy = torch.cat([outlet_x, y_flow], dim=1)
outlet_xy.requires_grad = True


def Loss(model):
    out = model(interior_xy)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # First derivatives
    u_xy = model.grad(u, interior_xy)
    u_x, u_y = u_xy[:, 0:1], u_xy[:, 1:2]

    v_xy = model.grad(v, interior_xy)
    v_x, v_y = v_xy[:, 0:1], v_xy[:, 1:2]

    p_xy = model.grad(p, interior_xy)
    p_x, p_y = p_xy[:, 0:1], p_xy[:, 1:2]

    # Continuity
    res_cont = u_x * L + v_y

    # Velocity gradient matrix J_ij = d(u_i)/d(x_j), where vel_0 = u, vel_1 = v
    J = [[u_x, u_y],
         [v_x, v_y]]

    # Assemble 2x2 stress tensor T[i][j]
    T = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            gamma = torch.abs(J[i][j] + J[j][i]) + 1e-12
            mu_eff = mu0 / (1.0 + (k * gamma) ** n)
            T[i][j] = mu_eff * gamma

    # Split via divergence: (dT_xx/dx + dT_xy/dy) and (dT_yx/dx + dT_yy/dy)
    dTxx_dx = model.grad(T[0][0], interior_xy)[:, 0:1]
    dTxy_dy = model.grad(T[0][1], interior_xy)[:, 1:2]

    dTyx_dx = model.grad(T[1][0], interior_xy)[:, 0:1]
    dTyy_dy = model.grad(T[1][1], interior_xy)[:, 1:2]

    mom_x_diff = -p_x * torch.full_like(p_x, L) + dTxx_dx * torch.full_like(p_x, L**2) + dTxy_dy
    mom_y_diff = -p_y + dTyx_dx * torch.full_like(p_y, L**2) + dTyy_dy

    pde_losses = torch.mean(mom_x_diff**2 + mom_y_diff**2 + res_cont**2)  # FIX 2: torch.mean

    # 1. No-Slip Walls (u=0, v=0)
    out_walls = model(walls_xy)
    loss_walls = torch.mean(out_walls[:, 0:2]**2)

    # 2. Inlet & Outlet Pressure
    out_inlet = model(inlet_xy)
    out_outlet = model(outlet_xy)
    p_outlet = out_outlet[:, 2:3]
    p_inlet = out_inlet[:, 2:3]
    p_inlet_bc = torch.full_like(p_inlet, Pnd_in)
    p_outlet_bc = torch.full_like(p_outlet, Pnd_out
                                )
    P_loss_total = torch.mean((p_inlet - p_inlet_bc)**2 + (p_outlet - p_outlet_bc)**2)
    terms = torch.stack([loss_walls,P_loss_total])
    bc_loss = torch.sqrt(torch.mean(terms**2) + 1e-12)
    
    ####conditions imposed
    """we have imposed the folowing conditons
        1. du/dx = 0
        2. dv/dy = 0
        3. v = 0
        4. dp/dy = 0
        6. dp/dx = k 
        we need to find these losses,this is the most important loss terms and have to me enforced strictly i.e more weightage
    """
    # --- Strict imposed condition losses ---
    loss_du_dx = torch.mean(u_x**2)
    loss_dv_dy = torch.mean(v_y**2)
    loss_v = torch.mean(v**2)
    loss_dp_dy = torch.mean(p_y**2)
    loss_dp_dx = torch.mean((p_x - torch.full_like(p_x,(Pnd_out-Pnd_in)/a))**2)
    terms = torch.stack([loss_du_dx, loss_dv_dy, loss_v, loss_dp_dy, loss_dp_dx])
    assumptions_loss = torch.sqrt(torch.mean(terms**2) + 1e-12)
    return pde_losses, bc_loss ,assumptions_loss

# ==========================================
# 1. NETWORK INITIALIZATION
# ==========================================
# Input: 2 dimensions (x, y)
# Output: 3 dimensions (u, v, p)
model = PINN_Network(input_dim=2, output_dim=3, num_hidden_layers=6, neurons_per_layer=64)
save_path = "pinn_cylinder_weights.pth"

if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path))
    print(f"Successfully loaded saved weights from {save_path}!")
else:
    print("No saved weights found. Starting from scratch with random Xavier weights.")

# ==========================================
# 2. OPTIMIZER & HYPERPARAMETERS
# ==========================================
if __name__ == "__main__":
    # Standard learning rate for PINNs. We use Adam to power through the rough loss landscape.
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 5000

    # ==========================================
    # 3. THE TRAINING LOOP
    # ==========================================
    print("Starting Adam Training Loop...")

    for epoch in range(epochs + 1):
        optimizer.zero_grad()

        pde_losses, bc_loss_total, assumption_losses = Loss(model)

        total_loss = 10 * pde_losses + bc_loss_total + 20 * assumption_losses

        total_loss.backward()
        optimizer.step()

        # Catalogue all active loss terms during monitoring
        if epoch % 500 == 0:
            print(f"Epoch {epoch:05d} | Total Loss: {total_loss.item():.4e}")
            print(
                f"  PDE        -> {pde_losses.item():.2e} (weighted: {(10 * pde_losses).item():.2e})"
            )
            print(f"  Boundary   -> {bc_loss_total.item():.2e}")
            print(
                f"  Assumptions-> {assumption_losses.item():.2e} (weighted: {(20 * assumption_losses).item():.2e})"
            )
            print("-" * 65)

    print("Training Complete!")
    # --- NEW: SAVE THE TRAINED WEIGHTS ---
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
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

#a = 1 #aspect ratio
L = h/l
#williamson model
k = 1.2 #consistency
n = 0.5 #powerlaw

class PINN_Network(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=3, neurons_per_layer=160):
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
ex = 1 / N  # aspect ratio into consideration

x_vals = torch.linspace(0 + ex, 1 - ex, N)
y_vals = torch.linspace(-1 + ey, 1 - ey, N)

X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")

X_flat = X.flatten().unsqueeze(1)
Y_flat = Y.flatten().unsqueeze(1)
interior_xy = torch.cat([X_flat, Y_flat], dim=1)
interior_xy.requires_grad = True  # FIX 1: required for autograd inside Loss()


# ==========================================
# BOUNDARY GENERATION
# ==========================================
N_b = 200

# 1. Top and Bottom Walls
x_wall = torch.linspace(0, 1, N_b).unsqueeze(1)

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

outlet_x = torch.full_like(y_flow, 1)
outlet_xy = torch.cat([outlet_x, y_flow], dim=1)
outlet_xy.requires_grad = True


def Loss(model):
    out = model(interior_xy)
    u, p = out[:, 0:1], out[:, 1:2]

    # First derivatives
    u_xy = model.grad(u, interior_xy)
    u_x, u_y = u_xy[:, 0:1], u_xy[:, 1:2]

    # v = 0 identically -> derivatives are identically 0 (no autograd call needed)
    v_x = torch.zeros_like(u_x)
    v_y = torch.zeros_like(u_y)

    # FIX: unpack both columns of p_xy
    p_xy = model.grad(p, interior_xy)
    p_x, p_y = p_xy[:, 0:1], p_xy[:, 1:2]
    p_xx = model.grad(p_x, interior_xy)[:, 0:1]

    # Continuity (since v_y = 0, this also enforces du/dx = 0)
    res_cont = u_x * L + v_y

    # Velocity gradient matrix
    J = [[u_x, u_y],
         [v_x, v_y]]

    # Assemble 2x2 stress tensor T[i][j]
    T = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            strain_rate = J[i][j] + J[j][i]
            gamma_mag = torch.abs(strain_rate) + 1e-12
            mu_eff = mu0 / (1.0 + (k * gamma_mag) ** n)
            T[i][j] = mu_eff * strain_rate

    # Split via divergence (T[1][1] == 0 identically, so dTyy_dy = 0)
    dTxx_dx = model.grad(T[0][0], interior_xy)[:, 0:1]
    dTxy_dy = model.grad(T[0][1], interior_xy)[:, 1:2]
    dTyx_dx = model.grad(T[1][0], interior_xy)[:, 0:1]
    dTyy_dy = torch.zeros_like(u_y)

    mom_x_diff = -p_x * L + dTxx_dx * (L**2) + dTxy_dy
    mom_y_diff = -p_y + dTyx_dx * (L**2) + dTyy_dy

    pde_losses = torch.mean(mom_x_diff**2 + mom_y_diff**2 + res_cont**2)

    # 1. No-Slip Walls: only u = 0 (column 0), NOT pressure (column 1)
    out_walls = model(walls_xy)
    loss_walls = torch.mean(out_walls[:, 0:1]**2)

    # 2. Inlet & Outlet Pressure: column 1
    out_inlet = model(inlet_xy)
    out_outlet = model(outlet_xy)
    p_inlet = out_inlet[:, 1:2]
    p_outlet = out_outlet[:, 1:2]
    P_loss_total = torch.mean((p_inlet - Pnd_in)**2 + (p_outlet - Pnd_out)**2)

    # 3. Strict assumptions: dp/dy = 0 and d2p/dx2 = 0
    loss_dp_dy = torch.mean(p_y**2)
    loss_dp_dx = torch.mean(p_xx**2)

    # Return primary PDE objective + individual constraint terms for ALM/KKT
    constraints = {
        "pde" : pde_losses,
        "walls": loss_walls,
        "pressure_bc": P_loss_total,
        "dp_dy": loss_dp_dy,
        "d2p_dx2": loss_dp_dx,
    }
    return  constraints

model = PINN_Network(input_dim=2, output_dim=2, num_hidden_layers=6, neurons_per_layer=64)
save_path = "pinn_channel_2out_weights.pth"

if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"Loaded saved weights from {save_path}")

# ==========================================
# 2. OPTIMIZER & HYPERPARAMETERS
# ==========================================
if __name__ == "__main__":
    weights = {
        "pde": 10.0,
        "walls": 1.0,
        "pressure_bc": 1.0,
        "dp_dy": 20.0,
        "d2p_dx2": 20.0,
    }

    def compute_weighted_loss(constraints):
        return sum(weights[k] * constraints[k] for k in weights)

    def log_losses(prefix, step, total_loss, constraints):
        print(f"[{prefix}] Step {step:05d} | Total Loss: {total_loss.item():.4e}")
        print(
            f"  PDE: {constraints['pde'].item():.2e} | "
            f"Walls: {constraints['walls'].item():.2e} | "
            f"P_BC: {constraints['pressure_bc'].item():.2e}"
        )
        print(
            f"  dp/dy: {constraints['dp_dy'].item():.2e} | "
            f"d2p/dx2: {constraints['d2p_dx2'].item():.2e}"
        )
        print("-" * 65)

    # ==========================================
    # STAGE 1: ADAM (GLOBAL EXPLORATION)
    # ==========================================
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 5000

    print("Starting Stage 1: Adam Training Loop...")

    for epoch in range(epochs + 1):
        optimizer.zero_grad()

        constraints = Loss(model)
        total_loss = compute_weighted_loss(constraints)

        total_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            log_losses("Adam", epoch, total_loss, constraints)

    # ==========================================
    # STAGE 2: L-BFGS (LOCAL FINE-TUNING)
    # ==========================================
    print("Starting Stage 2: L-BFGS Fine-Tuning...")

    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1000,
        max_eval=1250,
        history_size=50,
        tolerance_grad=1e-7,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe",
    )

    lbfgs_eval = [0]

    def closure():
        lbfgs_optimizer.zero_grad()

        constraints = Loss(model)
        total_loss = compute_weighted_loss(constraints)

        total_loss.backward()

        if lbfgs_eval[0] % 100 == 0:
            log_losses("L-BFGS", lbfgs_eval[0], total_loss, constraints)

        lbfgs_eval[0] += 1
        return total_loss

    lbfgs_optimizer.step(closure)

    print("Training Complete!")
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
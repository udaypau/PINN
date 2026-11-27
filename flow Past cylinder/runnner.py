import torch
import torch.nn as nn

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TraiN_the_model = True
Re = 100.0
"""
    2D Dimensionless Navier-Stokes Equations (Lid-Driven Cavity Flow):
    x-momentum:  u* (du*/dx*) + v* (du*/dy*) + dp*/dx* - (1/Re) (d^2u*/dx*^2 + d^2u*/dy*^2) = 0
    y-momentum:  u* (dv*/dx*) + v* (dv*/dy*) + dp*/dy* - (1/Re) (d^2v*/dx*^2 + d^2v*/dy*^2) = 0
    continuity:  du*/dx* + dv*/dy* = 0

    u* = u/U0, v* = v/U0, p* = p/(rho*U0^2)
    x* = x/L,  y* = y/L
    Re = rho*U0*L/mu

    b.cs: all walls u = v = 0
          at top wall u = 1, v = 0
          pressure anchor at (0, 0): p = 0
"""


class PINN_Network(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, num_hidden_layers=4, neurons_per_layer=64):
        super().__init__()
        layers = [nn.Linear(input_dim, neurons_per_layer), nn.SiLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(neurons_per_layer, output_dim))

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, xy):
        return self.network(xy)

    @staticmethod
    def grad(f, x):
        return torch.autograd.grad(
            f, x, grad_outputs=torch.ones_like(f), create_graph=True
        )[0]


N = 100
e = 1e-3
x_vals = torch.linspace(0 + e, 1 - e, N, device=device)  # bounds for x [0, 1]
y_vals = torch.linspace(0 + e, 1 - e, N, device=device)  # bounds for y [0, 1]

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
x_wall = torch.linspace(0, 1, N_b, device=device).unsqueeze(1)
x_top = torch.linspace(0 + e, 1 - e, N_b, device=device).unsqueeze(1)  # avoid top corner singularities

top_wall_y = torch.full_like(x_top, 1.0)
top_wall_xy = torch.cat([x_top, top_wall_y], dim=1)

bottom_wall_y = torch.full_like(x_wall, 0.0)
bottom_wall_xy = torch.cat([x_wall, bottom_wall_y], dim=1)

# side walls
y_wall = torch.linspace(0, 1, 2 * N_b, device=device).unsqueeze(1)

side_wall_x_l = torch.full_like(y_wall, 0.0)
side_wall_x_r = torch.full_like(y_wall, 1.0)

side_wall_xy_l = torch.cat([side_wall_x_l, y_wall], dim=1)
side_wall_xy_r = torch.cat([side_wall_x_r, y_wall], dim=1)
side_wall_xy = torch.cat([side_wall_xy_l, side_wall_xy_r], dim=0)

# Pressure anchor at (0, 0)
p0_xy = torch.tensor([[0.0, 0.0]], device=device)
# omitting autograd since the b.cs are not derivative based


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

    # Second derivatives
    u_xx = model.grad(u_x, interior_xy)[:, 0:1]
    u_yy = model.grad(u_y, interior_xy)[:, 1:2]

    v_xx = model.grad(v_x, interior_xy)[:, 0:1]
    v_yy = model.grad(v_y, interior_xy)[:, 1:2]

    # Continuity & Momentum Residuals
    res_cont = u_x + v_y
    res_xmom = u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
    res_ymom = u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)

    pde_losses = torch.mean(res_cont**2 + res_xmom**2 + res_ymom**2)

    # --- Boundary Conditions ---
    # Top wall (lid): u = 1, v = 0
    out_top = model(top_wall_xy)
    loss_top = torch.mean((out_top[:, 0:1] - 1.0)**2 + out_top[:, 1:2]**2)

    # Bottom wall: u = 0, v = 0
    out_bottom = model(bottom_wall_xy)
    loss_bottom = torch.mean(out_bottom[:, 0:1]**2 + out_bottom[:, 1:2]**2)

    # Side walls: u = 0, v = 0
    out_side = model(side_wall_xy)
    loss_side = torch.mean(out_side[:, 0:1]**2 + out_side[:, 1:2]**2)

    # Pressure anchor: p(0, 0) = 0
    loss_p0 = torch.mean(model(p0_xy)[:, 2:3]**2)

    bc_loss = loss_top + loss_bottom + loss_side + loss_p0
    return pde_losses, bc_loss


# ==========================================
# NETWORK INITIALIZATION & TRAINING
# ==========================================
model = PINN_Network().to(device)
save_path = "pinn_cavity_weights.pth"

if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"Loaded weights from {save_path}")

if __name__ == "__main__" and TraiN_the_model:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10000

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        pde_losses, bc_loss = Loss(model)
        total_loss = pde_losses + 10.0 * bc_loss
        total_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:05d} | Total: {total_loss.item():.4e} | PDE: {pde_losses.item():.2e} | BC: {bc_loss.item():.2e}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")



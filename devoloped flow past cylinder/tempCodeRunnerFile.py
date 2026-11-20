import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
"""
        p* = (p*h^2)/(mu0*L)  where mu0 = refrence values
        y*  = y/h                    h = height of pipe
        x*  = x/l                    l =  length of pipe
"""
Pnd_in = 2 
Pnd_out = 0 
h = 1 
L  = 10
mu = 0.1

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
    def grad(f, xy):        # <---------------------------------- Differential operator
        """Gradient of scalar field f (N,1) w.r.t. xy (N,2) -> (N,2): [d/dx*, d/dy*]"""
        return torch.autograd.grad(f, xy, grad_outputs=torch.ones_like(f),
                               create_graph=True)[0]
    

##########################################################################################################
#                                     GEOMETRY SECTION                                                   #
##########################################################################################################
# 1. Define the 80x80 Grid
N = 80
ey = h/N
ex = L/N
# 1. FIX: Shift X domain to [-1, 2] so the cylinder at (0,0) is fully submerged
x_vals = torch.linspace(0+ex, L-ex, N)  
y_vals = torch.linspace(-h+ey, h-ey, N)  

X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
X0 = torch.full_like(X,L/2)
y0 = torch.full_like(Y,0)

diameter = h / 2.0
radius = diameter / 2.0  

distance_sq = (X-X0)**2 + (Y-y0)**2
mask_circle = (distance_sq > radius**2).float()

# --- Extract Active Interior Points ---
X_flat = X.flatten().unsqueeze(1)
Y_flat = Y.flatten().unsqueeze(1)
mask_flat = mask_circle.flatten().unsqueeze(1)

xy_interior_full = torch.cat([X_flat, Y_flat], dim=1)
active_indices = (mask_flat == 1.0).squeeze()

# THIS is the tensor you pass to the PDE loss function
interior_xy = xy_interior_full[active_indices] 
interior_xy.requires_grad = True


# ==========================================
# BOUNDARY GENERATION
# ==========================================
N_b = 200 
R_cyl = 0.25

# 1. Top and Bottom Walls (x goes from  to  now)
x_wall = torch.linspace(0, L, N_b).unsqueeze(1)

top_wall_y = torch.full_like(x_wall, 1.0)
top_wall_xy = torch.cat([x_wall, top_wall_y], dim=1)

bottom_wall_y = torch.full_like(x_wall, -1.0)
bottom_wall_xy = torch.cat([x_wall, bottom_wall_y], dim=1)
# 3. The Cylinder Wall (Center is still 0,0)
theta = torch.linspace(0, 2 * math.pi, N_b).unsqueeze(1)
cyl_x = R_cyl * torch.cos(theta) + X0
cyl_y = R_cyl * torch.sin(theta) + y0
cylinder_wall_xy = torch.cat([cyl_x, cyl_y], dim=1)

# 1. No-Slip Walls (u=0, v=0)
walls_xy = torch.cat([top_wall_xy, bottom_wall_xy, cylinder_wall_xy], dim=0)
walls_xy.requires_grad = True


# 2. Inlet and Outlet
y_flow = torch.linspace(-h, h, N_b).unsqueeze(1)


inlet_x = torch.full_like(y_flow,0)
inlet_xy = torch.cat([inlet_x, y_flow], dim=1)
inlet_xy.requires_grad = True

outlet_x = torch.full_like(y_flow, L)
outlet_xy = torch.cat([outlet_x, y_flow], dim=1)
outlet_xy.requires_grad = True



def grad(f, xy):        # <---------------------------------- Differential operator
    """Gradient of scalar field f (N,1) w.r.t. xy (N,2) -> (N,2): [d/dx*, d/dy*]"""
    return torch.autograd.grad(f, xy, grad_outputs=torch.ones_like(f),
                               create_graph=True)[0]


def Loss(model):
    # Dynamic Viscosity (normalized or physical)
    
    
    out = model(interior_xy)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # First derivatives
    u_xy = grad(u, interior_xy)
    u_x, u_y = u_xy[:, 0:1], u_xy[:, 1:2]
    
    v_xy = grad(v, interior_xy)
    v_x, v_y = v_xy[:, 0:1], v_xy[:, 1:2]
    
    p_xy = grad(p, interior_xy)
    p_x, p_y = p_xy[:, 0:1], p_xy[:, 1:2]

    # Second derivatives
    u_xx = grad(u_x, interior_xy)[:, 0:1]
    u_yy = grad(u_y, interior_xy)[:, 1:2]
    
    v_xx = grad(v_x, interior_xy)[:, 0:1]
    v_yy = grad(v_y, interior_xy)[:, 1:2]

    # 2D Navier-Stokes Residuals
    res_cont = u_x + v_y
    
    # X-Momentum: (Convection) + (Pressure) - (Viscous Diffusion)
    res_momx = (u * u_x + v * u_y) + p_x -  mu*(u_xx + u_yy)
    
    # Y-Momentum: (Convection) + (Pressure) - (Viscous Diffusion)
    res_momy = (u * v_x + v * v_y) + p_y -  mu*(v_xx + v_yy)
    
    pde_losses = torch.mean(res_cont**2), torch.mean(res_momx**2), torch.mean(res_momy**2)
    # 1. No-Slip Walls (u=0, v=0)
    out_walls = model(walls_xy)
    loss_walls = torch.mean(out_walls[:, 0:2]**2) # Penalizes both u and v

    # 2. Inlet (Parabolic Profile: u = 1 - y^2, v = 0)
    out_inlet = model(inlet_xy)
    u_inlet, v_inlet = out_inlet[:, 0:1], out_inlet[:, 1:2]
    u_target = 1.0 - inlet_xy[:, 1:2]**2
    loss_inlet = torch.mean((u_inlet - u_target)**2) + torch.mean(v_inlet**2)
    
    # 3. pressure
    out_outlet = model(outlet_xy)
    p_outlet = out_outlet[:, 2:3]
    p_inlet = out_inlet[:,2:3]
    p_inlet_bc = torch.full_like(p_inlet,Pnd_in)
    p_outlet_bc = torch.full_like(p_outlet,Pnd_out)
    P_loss_total = torch.mean((p_inlet  - p_inlet_bc)**2 + (p_outlet- p_outlet_bc)**2)
    bc_loss =  loss_walls + loss_inlet + P_loss_total

    return pde_losses,bc_loss
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
# Standard learning rate for PINNs. We use Adam to power through the rough loss landscape.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 1000

# ==========================================
# 3. THE TRAINING LOOP (FIXED)
# ==========================================
print("Starting Adam Training Loop...")

for epoch in range(epochs + 1):
    optimizer.zero_grad()
    
    # Unpack the tuple returned by the Loss function
    pde_losses_tuple, bc_loss_total = Loss(model)
    l_c, l_mx, l_my = pde_losses_tuple
    
    # Apply weights (Sum the PDE losses first, then multiply)
    pde_loss_total = l_c + l_mx + l_my
    total_loss = (pde_loss_total * 10) + bc_loss_total
    
    total_loss.backward()
    optimizer.step()
    
    # Monitoring using the variables that actually exist in this scope
    if epoch % 500 == 0:
        print(f"Epoch {epoch:05d} | Total Loss: {total_loss.item():.4e}")
        print(f"  PDE   -> Cont: {l_c.item():.2e} | X-Mom: {l_mx.item():.2e} | Y-Mom: {l_my.item():.2e}")
        print(f"  Bound -> Total BC Loss: {bc_loss_total.item():.2e}")
        print("-" * 65)

print("Training Complete!")

# ==========================================
# 4. PLOTTING FUNCTION (FIXED)
# ==========================================
def plot_results(model):
    print("Generating flow visualization...")
    
    x_grid = np.linspace(0, 10, 300)
    y_grid = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    xy_plot_tensor = torch.tensor(
        np.column_stack((X.flatten(), Y.flatten())), 
        dtype=torch.float32
    )
    
    model.eval()
    with torch.no_grad():
        preds = model(xy_plot_tensor)
        
    u_pred = preds[:, 0].numpy().reshape(X.shape)
    v_pred = preds[:, 1].numpy().reshape(X.shape)
    p_pred = preds[:, 2].numpy().reshape(X.shape)
    
    R_cyl = 0.25
    # FIX: Shift the plotting mask to X = 5 (since L=10 and X0=L/2)
    inside_cylinder = ((X - 5)**2 + Y**2) <= R_cyl**2
    
    u_pred[inside_cylinder] = np.nan
    v_pred[inside_cylinder] = np.nan
    p_pred[inside_cylinder] = np.nan
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    contour = ax.contourf(X, Y, p_pred, levels=50, cmap='jet', alpha=0.8)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Pressure (p)')
    
    step = 10 
    ax.quiver(X[::step, ::step], Y[::step, ::step], 
              u_pred[::step, ::step], v_pred[::step, ::step], 
              color='black', scale=25, headwidth=4, alpha=0.7)
    
    # FIX: Draw the physical circle at x=5
    cylinder_patch = patches.Circle((5, 0), R_cyl, color='darkgray', ec='black', lw=1.5, zorder=5)
    ax.add_patch(cylinder_patch)
    
    ax.set_aspect('equal') 
    ax.set_xlim([0, 10])
    ax.set_ylim([-1, 1])
    ax.set_title('PINN Steady Flow: Velocity Vectors over Pressure Contours')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.show()

plot_results(model)
print("Training Complete!")

# --- NEW: SAVE THE TRAINED WEIGHTS ---
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")
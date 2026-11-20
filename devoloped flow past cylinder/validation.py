import os
import torch 
from Runner import PINN_Network
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
Pnd_in = 2 
Pnd_out = 0 
h = 1 
L  = 10
mu = 0.1
save_path = "pinn_cylinder_weights.pth"
plot_path = "pinn_cylinder_results.png"
if os.path.exists(save_path):
    model = PINN_Network(input_dim=2, output_dim=3, num_hidden_layers=6, neurons_per_layer=64)
    model.load_state_dict(torch.load(save_path,weights_only=True))
    print(f"Successfully loaded saved weights from {save_path}!")
    print("Generating flow visualization...")
    
    x_grid = np.linspace(0, L, 300)
    y_grid = np.linspace(-h, h, 200)
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
    vel_mag = np.sqrt(u_pred**2 + v_pred**2)
    
    R_cyl = 0.25
    cx = L / 2.0
    inside_cylinder = ((X - cx)**2 + Y**2) <= R_cyl**2
    
    u_pred[inside_cylinder] = np.nan
    v_pred[inside_cylinder] = np.nan
    p_pred[inside_cylinder] = np.nan
    vel_mag[inside_cylinder] = np.nan
    
    # Create 2 stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # --- Graph 1: Pressure Field ---
    contour = ax1.contourf(X, Y, p_pred, levels=50, cmap='jet')
    cbar1 = fig.colorbar(contour, ax=ax1)
    cbar1.set_label('Pressure (p)')
    ax1.add_patch(patches.Circle((cx, 0), R_cyl, color='darkgray', ec='black', lw=1.5, zorder=5))
    ax1.set_aspect('equal')
    ax1.set_xlim([0, L])
    ax1.set_ylim([-h, h])
    ax1.set_title('Pressure Contours')
    ax1.set_ylabel('y')
    plt.tight_layout()
    # --- Graph 2: Velocity Vectors ---
    u_stream = np.where(inside_cylinder,0.0,u_pred)
    v_stream = np.where(inside_cylinder,0.0,v_pred)
    strm = ax2.streamplot(
        X, Y, u_stream, v_stream,
        color=vel_mag, cmap='plasma', density=2, linewidth=0.5, arrowsize=0.5
    )
    fig.colorbar(strm.lines, ax=ax2, label='Velocity Magnitude |V|')
   

    ax2.add_patch(patches.Circle((cx, 0), R_cyl, color='darkgray', ec='black', lw=1.5, zorder=5))
    ax2.set_aspect('equal')
    ax2.set_xlim([0, L])
    ax2.set_ylim([-h, h])
    ax2.set_title('Flow Streamlines')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.tight_layout()
    fig.savefig(plot_path,dpi=300, bbox_inches='tight')
    plt.show()
    
    
else:
    print("No saved weights found. Starting from scratch with random Xavier weights.")



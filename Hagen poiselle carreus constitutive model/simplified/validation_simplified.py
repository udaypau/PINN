import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from runner_simplified import PINN_Network

save_path = "pinn_channel_2out_weights.pth"
plot_path = "pinn_channel_evaluation.png"

L = 1
h = 1.0

if os.path.exists(save_path):
    model = PINN_Network(input_dim=2, output_dim=2, num_hidden_layers=6, neurons_per_layer=64)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    print(f"Loaded weights from {save_path}")

    # Uniform grid
    Nx, Ny = 100, 80
    x_vals = torch.linspace(0.0, L, Nx)
    y_vals = torch.linspace(-h, h, Ny)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="xy")

    xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        preds = model(xy_grid)

    X_np, Y_np = X.numpy(), Y.numpy()
    u_pred = preds[:, 0].numpy().reshape(X.shape)
    p_pred = preds[:, 1].numpy().reshape(X.shape)

    # Plot u and p
    fig, (ax_u, ax_p) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    c_u = ax_u.contourf(X_np, Y_np, u_pred, levels=50, cmap="coolwarm")
    fig.colorbar(c_u, ax=ax_u, label="Axial Velocity (u)")
    ax_u.set_title("Axial Velocity Field u(x, y)")
    ax_u.set_ylabel("y")

    c_p = ax_p.contourf(X_np, Y_np, p_pred, levels=50, cmap="jet")
    fig.colorbar(c_p, ax=ax_p, label="Pressure (p)")
    ax_p.set_title("Pressure Field p(x, y)")
    ax_p.set_xlabel("x")
    ax_p.set_ylabel("y")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved evaluation plot to {plot_path}")
    plt.show()

else:
    print(f"No saved weights found at {save_path}.")
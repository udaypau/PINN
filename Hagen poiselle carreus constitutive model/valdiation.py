import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from runner import PINN_Network, Loss

Pnd_in = 2
Pnd_out = 0
h = 1
L = 10
mu = 0.1
a = 10

save_path = "pinn_cylinder_weights.pth"
plot_path = "pinn_channel_evaluation.png"

if os.path.exists(save_path):
    model = PINN_Network(
        input_dim=2, output_dim=3, num_hidden_layers=6, neurons_per_layer=64
    )
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"Successfully loaded saved weights from {save_path}!")

    # ------------------------------------------
    # 4. EVALUATE LOSS, STACK & COMPUTE RMS
    # ------------------------------------------
    pde_loss, bc_loss, assumption_loss = Loss(model)

    stacked_losses = torch.stack([
        pde_loss.detach(),
        bc_loss.detach(),
        assumption_loss.detach(),
    ])
    rms_loss = torch.sqrt(torch.mean(stacked_losses ** 2) + 1e-12)


    # ------------------------------------------
    # GRID EVALUATION FOR PLOTTING
    # ------------------------------------------
    print("Generating flow visualization...")
    x_grid = np.linspace(0, a, 300)
    y_grid = np.linspace(-h, h, 200)
    X, Y = np.meshgrid(x_grid, y_grid)

    xy_plot_tensor = torch.tensor(
        np.column_stack((X.flatten(), Y.flatten())),
        dtype=torch.float32,
    )

    model.eval()
    with torch.no_grad():
        preds = model(xy_plot_tensor)

    u_pred = preds[:, 0].numpy().reshape(X.shape)
    v_pred = preds[:, 1].numpy().reshape(X.shape)
    p_pred = preds[:, 2].numpy().reshape(X.shape)
    vel_mag = np.sqrt(u_pred ** 2 + v_pred ** 2)

    # ------------------------------------------
    # 1-4. PLOTS: PRESSURE, u, v, STREAMLINES & RMS ERROR
    # ------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(12, 17))
    ax_p, ax_u, ax_v, ax_s, ax_rms = axes

    # 1. Pressure Contours
    c_p = ax_p.contourf(X, Y, p_pred, levels=50, cmap="jet")
    fig.colorbar(c_p, ax=ax_p, label="Pressure (p)")
    ax_p.set_title("1. Pressure Field p(x, y)")
    ax_p.set_xlabel("x")
    ax_p.set_ylabel("y")
    ax_p.set_xlim([0, a])
    ax_p.set_ylim([-h, h])

    # 2a. Velocity u Contours
    c_u = ax_u.contourf(X, Y, u_pred, levels=50, cmap="coolwarm")
    fig.colorbar(c_u, ax=ax_u, label="Axial Velocity (u)")
    ax_u.set_title("2a. Axial Velocity u(x, y)")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("y")
    ax_u.set_xlim([0, a])
    ax_u.set_ylim([-h, h])

    # 2b. Velocity v Contours
    c_v = ax_v.contourf(X, Y, v_pred, levels=50, cmap="coolwarm")
    fig.colorbar(c_v, ax=ax_v, label="Transverse Velocity (v)")
    ax_v.set_title("2b. Transverse Velocity v(x, y)")
    ax_v.set_xlabel("x")
    ax_v.set_ylabel("y")
    ax_v.set_xlim([0, a])
    ax_v.set_ylim([-h, h])

    # 3. Streamlines
    strm = ax_s.streamplot(
        X, Y, u_pred, v_pred,
        color=vel_mag, cmap="plasma", density=1.6, linewidth=1.2, arrowsize=1.2
    )
    fig.colorbar(strm.lines, ax=ax_s, label="Speed |V|")
    ax_s.set_title("3. Flow Streamlines")
    ax_s.set_xlabel("x")
    ax_s.set_ylabel("y")
    ax_s.set_xlim([0, a])
    ax_s.set_ylim([-h, h])

    # 4. RMS Error / Loss Breakdown Plot
    loss_labels = ["PDE Loss", "BC Loss", "Assumption Loss", "Combined RMS"]
    loss_values = [
        pde_loss.item(),
        bc_loss.item(),
        assumption_loss.item(),
        rms_loss.item(),
    ]
    bar_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

    bars = ax_rms.bar(loss_labels, loss_values, color=bar_colors, width=0.45)
    ax_rms.set_yscale("log")
    ax_rms.set_title("4. Loss Components & Combined RMS Error (Log Scale)")
    ax_rms.set_ylabel("Loss Magnitude")
    ax_rms.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, loss_values):
        ax_rms.text(
            bar.get_x() + bar.get_width() / 2.0,
            val * 1.15,
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved evaluation figure to {plot_path}")
    plt.show()

else:
    print(f"No saved weights found at {save_path}.")
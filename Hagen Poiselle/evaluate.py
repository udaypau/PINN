import yaml
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. EXACT MODEL CLASS FROM HagenPoisuelleV2
# =============================================================================
ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}
def _pick(table, name): return table.get(str(name).lower(), nn.SiLU)

class NavierStokesPINN_SIMPLE(nn.Module):
    def __init__(self, in_features=2, num_layers=9, num_hidden=160, out_features=2, activation="silu"):
        super().__init__()
        act_func = _pick(ACTIVATIONS, activation)
        layers = [nn.Linear(in_features, num_hidden), act_func()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(act_func())
        layers.append(nn.Linear(num_hidden, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        if isinstance(xy, (list, tuple)): xy = torch.cat(xy, dim=1)
        return self.net(xy)

# =============================================================================
# 2. LOAD YAML, SETUP MODEL & LOAD WEIGHTS
# =============================================================================
cfg_path = "params.yaml"
if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
    cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

model = NavierStokesPINN_SIMPLE(
    num_layers=cfg["num_layers"],
    num_hidden=cfg["num_hidden"],
    activation=cfg["activation"]
).to(device)

# Load weights (Ensure "best_model.pt" was saved, or point to "runs/<run_name>/weights.pt")
state = torch.load("best_model.pt", map_location=device)
# Extract state_dict if bundled, otherwise use directly
model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
model.eval()

# =============================================================================
# 3. READ CSV & NON-DIMENSIONALIZE
# =============================================================================
df = pd.read_csv("results.csv")
if 'Points:2' in df.columns:
    df = df[df['Points:2'] == df['Points:2'].min()]  # Slice 2D midplane
# Physics scales
H = cfg["H"]
L = cfg["L"]
mu = cfg["mu"]
u0 = cfg["u0"]

x_raw, y_raw = df['Points:0'].values, df['Points:1'].values
x_star = x_raw/L
y_star = y_raw/H



u_true = df['U:0'].values / u0
p_true =(df['p'].values * (H ** 2)) / (mu * u0 * L)

# =============================================================================
# 4. EVALUATE PREDICTIONS & AUTOGRAD GRADIENTS
# =============================================================================
xy_tensor = torch.tensor(np.stack([x_star, y_star], axis=1), dtype=torch.float32, device=device, requires_grad=True)

pred = model(xy_tensor)
u_pred, p_pred = pred[:, 0:1], pred[:, 1:2]


u_pred_np = u_pred.detach().cpu().numpy().flatten()
p_pred_np = p_pred.detach().cpu().numpy().flatten()

# Metrics
print(f"Absolute RMS Error (u*): {np.sqrt(np.mean((u_pred_np - u_true)**2)):.4e}")
print(f"Absolute RMS Error (p*): {np.sqrt(np.mean((p_pred_np - p_true)**2)):.4e}")

# =============================================================================
# 5. PLOT & SAVE
# =============================================================================
# =============================================================================
# 5. PLOT & SAVE (6 Plots: True, Predicted, Error for U and P)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- ROW 1: VELOCITY U ---
# 1. True U
sc0 = axes[0, 0].scatter(x_star, y_star, c=u_true, cmap='viridis', s=20)
axes[0, 0].set_title(r"True Velocity $u^*_{\mathrm{true}}$")
fig.colorbar(sc0, ax=axes[0, 0])

# 2. Predicted U
sc1 = axes[0, 1].scatter(x_star, y_star, c=u_pred_np, cmap='viridis', s=20)
axes[0, 1].set_title(r"Predicted Velocity $u^*_{\mathrm{pred}}$")
fig.colorbar(sc1, ax=axes[0, 1])

# 3. Absolute Error U
sc2 = axes[0, 2].scatter(x_star, y_star, c=np.abs(u_pred_np - u_true), cmap='plasma', s=20)
axes[0, 2].set_title(r"Absolute Error $|u^*_{\mathrm{pred}} - u^*_{\mathrm{true}}|$")
fig.colorbar(sc2, ax=axes[0, 2])


# --- ROW 2: PRESSURE P ---
# 4. True P
sc3 = axes[1, 0].scatter(x_star, y_star, c=p_true, cmap='coolwarm', s=20)
axes[1, 0].set_title(r"True Pressure $p^*_{\mathrm{true}}$")
fig.colorbar(sc3, ax=axes[1, 0])

# 5. Predicted P
sc4 = axes[1, 1].scatter(x_star, y_star, c=p_pred_np, cmap='coolwarm', s=20)
axes[1, 1].set_title(r"Predicted Pressure $p^*_{\mathrm{pred}}$")
fig.colorbar(sc4, ax=axes[1, 1])

# 6. Absolute Error P
sc5 = axes[1, 2].scatter(x_star, y_star, c=np.abs(p_pred_np - p_true), cmap='plasma', s=20)
axes[1, 2].set_title(r"Absolute Error $|p^*_{\mathrm{pred}} - p^*_{\mathrm{true}}|$")
fig.colorbar(sc5, ax=axes[1, 2])

plt.tight_layout()
plt.savefig("evaluation_maps.png", dpi=200)
print("Saved evaluation_maps.png")
plt.show()



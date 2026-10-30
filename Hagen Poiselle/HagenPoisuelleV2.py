################################################################################
# Hagen-Poiseuille PINN  -  all parameters come from params.yaml               #
# Usage:  python hagen_poiseuille.py params.yaml                               #
################################################################################
import sys
import time
import copy
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--------------------------------------------------------------------------------#
# 0. READ PARAMETERS FROM YAML (unpack everything here, once)                    #
#--------------------------------------------------------------------------------#
cfg_path = "params.yaml"
if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
    cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

run_name    = cfg["run_name"]
seed        = cfg["seed"]
N           = cfg["n_points"]

NUM_LAYERS  = cfg["num_layers"]
NUM_HIDDEN  = cfg["num_hidden"]
ACTIVATION  = cfg["activation"]
INIT_WEIGHT = cfg["init_weight"]
INIT_BIAS   = cfg["init_bias"]

P_in        = float(cfg["p_in"])
P_out       = float(cfg["p_out"])
W           = cfg["weights"]

ADAM_LR      = float(cfg["adam_lr"])
MIN_LR       = float(cfg["min_lr"])
LR_FACTOR    = float(cfg["lr_factor"])
LR_PATIENCE  = cfg["lr_patience"]
SWITCH_LOSS  = float(cfg["switch_loss"])
MAX_ADAM     = cfg["max_adam"]
LBFGS_ROUNDS = cfg["lbfgs_rounds"]
LBFGS_ITERS  = cfg["lbfgs_iters"]
TOL          = float(cfg["tol"])
LOG_EVERY    = cfg["log_every"]

n_data      = cfg["n_data"]
data_seed   = cfg["data_seed"]

outdir = Path("runs") / run_name
outdir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(seed)
#---------------------------------------------------------------------------------#
# MODEL                                                                           #
#---------------------------------------------------------------------------------#
ACTIVATIONS = {
    # smooth: second derivatives exist and are nonzero (good for PDE residuals)
    "tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "mish": nn.Mish,
    "softplus": nn.Softplus, "sigmoid": nn.Sigmoid, "softsign": nn.Softsign,
    "tanhshrink": nn.Tanhshrink, "logsigmoid": nn.LogSigmoid,
    # second derivative has a jump at one point
    "elu": nn.ELU, "celu": nn.CELU, "selu": nn.SELU,
    # piecewise linear / flat: second derivative ~ 0 (mainly for comparison)
    "relu": nn.ReLU, "relu6": nn.ReLU6, "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU, "rrelu": nn.RReLU, "hardtanh": nn.Hardtanh,
    "hardswish": nn.Hardswish, "hardsigmoid": nn.Hardsigmoid,
    "hardshrink": nn.Hardshrink, "softshrink": nn.Softshrink,
}
 
WEIGHT_INITS = {
    "xavier_normal":   nn.init.xavier_normal_,
    "xavier_uniform":  nn.init.xavier_uniform_,
    "kaiming_normal":  nn.init.kaiming_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "normal":          lambda w: nn.init.normal_(w, mean=0.0, std=0.1),
}
BIAS_INITS = {
    "zeros":  nn.init.zeros_,
    "normal": lambda b: nn.init.normal_(b, mean=0.0, std=0.01),
}
 
 
def _pick(table, name, what):
    """Look up `name` (case-insensitive) in `table` with a readable error."""
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unknown {what} '{name}'. Available: {sorted(table)}")
    return table[key]
 
 
class NavierStokesPINN_SIMPLE(nn.Module):
    # `activation` is the NAME of the activation (string from the YAML)
    def __init__(self, in_features=2, num_layers=9, num_hidden=160, out_features=2,
                 activation="silu",
                 init_type_weight="xavier_normal", init_type_bias="zeros"):
        super().__init__()
        activation_func = _pick(ACTIVATIONS, activation, "activation")
        weight_init = _pick(WEIGHT_INITS, init_type_weight, "weight init")
        bias_init = _pick(BIAS_INITS, init_type_bias, "bias init")
 
        layers = [nn.Linear(in_features, num_hidden), activation_func()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(activation_func())
        layers.append(nn.Linear(num_hidden, out_features))
        self.net = nn.Sequential(*layers)
 
        for m in self.net:
            if isinstance(m, nn.Linear):
                weight_init(m.weight)
                bias_init(m.bias)
 
    def forward(self, input_para):
        # If input is a list/tuple of tensors [x, y], concatenate to (N, in_features)
        if isinstance(input_para, (list, tuple)):
            xy = torch.cat(input_para, dim=1)
        else:
            xy = input_para
        return self.net(xy)
 

#-------------------------------------------------------------------------------------------#
# Geometric sampling section (corners belong to the walls)
#-------------------------------------------------------------------------------------------#
x = torch.linspace(0.0, 1.0, N)
y = torch.linspace(-1.0, 1.0, N)
X, Y = torch.meshgrid(x, y, indexing="ij")
xy_all = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

mask_b_c = ((Y == -1) | (Y == 1)).reshape(-1)
mask_inlet = (((X == 0) | (X == 1)) & (Y > 0) & (Y < 1)).reshape(-1)

xy_bnd_bc = xy_all[mask_b_c].to(device)                    # walls (y*=0,1)
xy_bnd_inlet = xy_all[mask_inlet].to(device)               # x*=0 and x*=1 edges
xy_int = xy_all[~(mask_b_c | mask_inlet)].to(device).requires_grad_(True)

# Split inlet / outlet
xy_in = xy_bnd_inlet[xy_bnd_inlet[:, 0] == 0.0]
xy_out = xy_bnd_inlet[xy_bnd_inlet[:, 0] == 1.0]


#--------------------------------------------------------------------------------#
# EXACT SOLUTION + RANDOM TRAINING SET (used for the RMS in the progress report) #
#--------------------------------------------------------------------------------#
def exact_solution(xy, P_in=0.0, P_out=-2.0):
    """
        G* = -dp*/dx* = P_in - P_out
        u*(x*, y*) = (G*/2) y* (1 - y*)
        p*(x*, y*) = P_in + (P_out - P_in) x*
        v*         = 0   (not a network output, so it is not returned)
    """
    x, y = xy[:, 0:1], xy[:, 1:2]
    G = P_in - P_out
    u = 0.5 * G * y * (1.0 - y)
    p = P_in + (P_out - P_in) * x
    return u, p


def make_training_set(n_data=500, P_in=0.0, P_out=-2.0, seed=0):
    """Random points in [0,1]x[0,1] with exact u*, p*. Returns xy, u, p on device."""
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n_data, 2, generator=g)
    u, p = exact_solution(xy, P_in, P_out)
    return xy.to(device), u.to(device), p.to(device)


def data_loss(model, xy_data, u_data, p_data, eps=1e-20):
    """RMS error of the network vs the training set, for u* and p*."""
    out = model(xy_data)
    rms_u = torch.sqrt(torch.mean((out[:, 0:1] - u_data) ** 2) + eps)
    rms_p = torch.sqrt(torch.mean((out[:, 1:2] - p_data) ** 2) + eps)
    return rms_u, rms_p


#-----------------------------------------------------------------------------
# LOSS EVALUATOR MODULES
#-----------------------------------------------------------------------------
"""
Non-dimensional Hagen-Poiseuille flow between plates (PINN loss section)

Scales:
    x* = x/x0,  y* = y/h
    u* = u/u0,  u0 = 1 (arbitrary)
    p* = p h^2 / (mu u0 x0)

Network: (x*, y*) -> (u*, p*)

Non-dimensional equations (v = 0, fully developed):
    continuity   : du*/dx*                      = 0
    x-momentum   : dp*/dx* - d2u*/dy*2          = 0
    y-momentum   : dp*/dy*                      = 0
"""
def grad(f, xy):        # <---------------------------------- Differential operator
    """Gradient of scalar field f (N,1) w.r.t. xy (N,2) -> (N,2): [d/dx*, d/dy*]"""
    return torch.autograd.grad(f, xy, grad_outputs=torch.ones_like(f),
                               create_graph=True)[0]


def Loss(model, xy):
    out = model(xy)
    u, p = out[:, 0:1], out[:, 1:2]

    du = grad(u, xy)
    u_x, u_y = du[:, 0:1], du[:, 1:2]
    u_yy = grad(u_y, xy)[:, 1:2]

    dp = grad(p, xy)
    p_x, p_y = dp[:, 0:1], dp[:, 1:2]
    p_xx = grad(p_x, xy)[:, 0:1]

    res_cont = u_x               # continuity: du*/dx* = 0
    res_momx = p_x - u_yy        # x-momentum: dp*/dx* = d2u*/dy*2
    res_momy = p_y               # y-momentum: dp*/dy* = 0
    res_constP = p_xx            # constant pressure gradient: d2p*/dx*2 = 0
    return (torch.mean(res_cont ** 2),
            torch.mean(res_momx ** 2),
            torch.mean(res_momy ** 2),
            torch.mean(res_constP ** 2))


def bc_loss(model, xy_bnd_bc, xy_in, xy_out, P_in, P_out):
    # No-slip on the walls: u* = 0
    loss_wall = torch.mean(model(xy_bnd_bc)[:, 0:1] ** 2)
    # Pressure fixed at inlet and outlet (drives the flow)
    loss_pin = torch.mean((model(xy_in)[:, 1:2] - P_in) ** 2)
    loss_pout = torch.mean((model(xy_out)[:, 1:2] - P_out) ** 2)
    return loss_wall, loss_pin + loss_pout


#--------------------------------------------------------------------------------------#
# TRAINING LOOP                                                                        #
#--------------------------------------------------------------------------------------#
model = NavierStokesPINN_SIMPLE(in_features=2, num_layers=NUM_LAYERS,
                                num_hidden=NUM_HIDDEN, out_features=2,
                                activation=ACTIVATION,          # string from YAML
                                init_type_weight=INIT_WEIGHT,
                                init_type_bias=INIT_BIAS).to(device)

def compute_losses(model):
    """
    weighted : used ONLY for backprop / optimisation (uses W)
    plain    : unweighted sum of all terms, used for monitoring, LR
               scheduling, best-model selection and the Adam->L-BFGS switch
    parts    : individual unweighted terms
    """
    l_c, l_mx, l_my, l_cp = Loss(model, xy_int)
    l_wall, l_pres = bc_loss(model, xy_bnd_bc, xy_in, xy_out, P_in, P_out)
    weighted = (W["cont"] * l_c + W["momx"] * l_mx + W["momy"] * l_my
                + W["constP"] * l_cp + W["wall"] * l_wall + W["pres"] * l_pres)
    plain = l_c + l_mx + l_my + l_cp + l_wall + l_pres
    parts = dict(cont=l_c.item(), momx=l_mx.item(), momy=l_my.item(),
                 constP=l_cp.item(), wall=l_wall.item(), pres=l_pres.item())
    return weighted, plain, parts


history = []                       # unweighted loss history
best_loss, best_state, best_where = float("inf"), None, ""


def log(tag, it, plain, parts, t0):
    print(f"[{tag} {it:6d}] loss={plain:.3e} | "
          + " ".join(f"{k}={v:.1e}" for k, v in parts.items())
          + f" | {time.time() - t0:.0f}s")


# ---- Stage 1: Adam with plateau-based LR reduction ---------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
    min_lr=MIN_LR)

adam_stop = f"reached MAX_ADAM = {MAX_ADAM}"
t0 = time.time()
for ep in range(MAX_ADAM + 1):
    optimizer.zero_grad()
    weighted, plain, parts = compute_losses(model)
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    plain_val = plain.item()
    history.append(plain_val)
    scheduler.step(plain_val)                  # LR decays on unweighted loss

    if plain_val < best_loss:
        best_loss, best_state = plain_val, copy.deepcopy(model.state_dict())
        best_where = f"Adam epoch {ep}"

    if ep % LOG_EVERY == 0:
        log("Adam", ep, plain_val, parts, t0)
        print(f"             lr={optimizer.param_groups[0]['lr']:.2e}")

    if plain_val < SWITCH_LOSS:
        adam_stop = f"loss < SWITCH_LOSS ({SWITCH_LOSS:.0e})"
        print(f"\nUnweighted loss < {SWITCH_LOSS:.0e} at epoch {ep}: "
              f"switching to L-BFGS")
        break
    if optimizer.param_groups[0]["lr"] <= MIN_LR * 1.0001 and ep > LR_PATIENCE:
        adam_stop = "learning rate reached MIN_LR"
        print(f"\nLR reached minimum at epoch {ep}: switching to L-BFGS")
        break

adam_iters = ep + 1

# ---- Stage 2: L-BFGS in rounds, restarted from the best weights --------------
model.load_state_dict(best_state)
stall = 0
lbfgs_rounds_done, lbfgs_iters_done = 0, 0
for rnd in range(LBFGS_ROUNDS):
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS,
                              tolerance_grad=1e-14, tolerance_change=1e-16,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        weighted, _, _ = compute_losses(model)
        weighted.backward()
        return weighted

    lbfgs.step(closure)
    lbfgs_rounds_done += 1
    lbfgs_iters_done += lbfgs.state_dict()["state"][0]["n_iter"]

    _, plain, parts = compute_losses(model)
    plain_val = plain.item()
    history.append(plain_val)
    log("L-BFGS round", rnd, plain_val, parts, t0)

    if plain_val < best_loss:
        stall = stall + 1 if plain_val > 0.99 * best_loss else 0
        best_loss, best_state = plain_val, copy.deepcopy(model.state_dict())
        best_where = f"L-BFGS round {rnd}"
    else:
        stall += 1
        model.load_state_dict(best_state)      # discard a worse round

    if best_loss < TOL or stall >= 3:
        break

training_time = time.time() - t0
model.load_state_dict(best_state)
_, plain, best_parts = compute_losses(model)

#--------------------------------------------------------------------------------#
# PROGRESS REPORT                                                                #
#--------------------------------------------------------------------------------#
model.eval()
xy_d, u_d, p_d = make_training_set(n_data, P_in, P_out, data_seed)
with torch.no_grad():
    rms_u, rms_p = data_loss(model, xy_d, u_d, p_d)

report = [
    "=" * 60,
    f"PROGRESS REPORT  -  {run_name}",
    "=" * 60,
    f"Device                  : {device}",
    f"Network                 : {NUM_LAYERS} layers x {NUM_HIDDEN} units, {cfg['activation']}",
    f"Pressure bounds (p*)    : p_in = {P_in}, p_out = {P_out}",
    "-" * 60,
    f"Adam iterations         : {adam_iters}  (stopped: {adam_stop})",
    f"L-BFGS rounds           : {lbfgs_rounds_done}",
    f"L-BFGS iterations       : {lbfgs_iters_done}",
    f"TOTAL iterations        : {adam_iters + lbfgs_iters_done}",
    f"Time taken              : {training_time:.1f} s",
    "-" * 60,
    f"Best case found at      : {best_where}",
    f"Best unweighted loss    : {best_loss:.3e}",
    "Loss terms at best case :",
    *[f"    {k:7s} = {v:.3e}" for k, v in best_parts.items()],
    "-" * 60,
    f"RMS vs exact solution ({n_data} random points):",
    f"    u* = {rms_u.item():.3e}",
    f"    p* = {rms_p.item():.3e}",
    "=" * 60,
]
print("\n" + "\n".join(report))
with open(outdir / "progress_report.txt", "w") as f:
    f.write("\n".join(report) + "\n")

#--------------------------------------------------------------------------------#
# RESULTS                                                                        #
#--------------------------------------------------------------------------------#
with torch.no_grad():
    out = model(xy_all.to(device)).cpu().numpy()
U = out[:, 0].reshape(N, N)
P = out[:, 1].reshape(N, N)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].semilogy(history)
ax[0].set_title("Unweighted loss"); ax[0].set_xlabel("iteration / round")
c1 = ax[1].contourf(X.numpy(), Y.numpy(), U, 50, cmap="viridis")
ax[1].set_title("u*"); ax[1].set_xlabel("x*"); ax[1].set_ylabel("y*")
fig.colorbar(c1, ax=ax[1])
c2 = ax[2].contourf(X.numpy(), Y.numpy(), P, 50, cmap="coolwarm")
ax[2].set_title("p*"); ax[2].set_xlabel("x*")
fig.colorbar(c2, ax=ax[2])
plt.tight_layout()
plt.savefig(outdir / "pinn_results.png", dpi=150)

torch.save(best_state, outdir / "weights.pt")
torch.save(history, outdir / "history.pt")
print(f"Results saved in {outdir}/")

# Save a direct copy in the root folder for the evaluation script
torch.save(best_state, "best_model.pt")
################################################################################
# Library containing all useful fucntion for hagen poiseele flow visualization #
################################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = dict(cont=1.0, momx=1.0, momy=1.0,constP=1, wall=10.0, pres=10.0)
#---------------------------------------------------------------------------------#
# MODEL INITIALIZER CONTAINING VARIOUS ARCHITECTURAL BLUEPRINTS                   #
#---------------------------------------------------------------------------------#
class NavierStokesPINN_SIMPLE(nn.Module):
    def __init__(self, in_features=2, num_layers=9, num_hidden=160, out_features=2,
                 activation_func=nn.SiLU,
                 init_type_weight="xavier_normal", init_type_bias="zeros"):
        super().__init__()
        layers = [nn.Linear(in_features, num_hidden), activation_func()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(activation_func())

        layers.append(nn.Linear(num_hidden, out_features))
        self.net = nn.Sequential(*layers)

        weight_init = {
            "xavier_normal":  nn.init.xavier_normal_,
            "xavier_uniform": nn.init.xavier_uniform_,
            "kaiming_normal": nn.init.kaiming_normal_,
            "kaiming_uniform": nn.init.kaiming_uniform_,
            "normal":         lambda w: nn.init.normal_(w, mean=0.0, std=0.1),
        }
        bias_init = {
            "zeros":  nn.init.zeros_,
            "normal": lambda b: nn.init.normal_(b, mean=0.0, std=0.01),
        }

        for m in self.net:
            if isinstance(m, nn.Linear):
                weight_init[init_type_weight](m.weight)
                bias_init[init_type_bias](m.bias)

    def forward(self, input_para):
        # If input is a list/tuple of tensors [x, y], concatenate to (N, in_features)
        if isinstance(input_para, (list, tuple)):
            xy = torch.cat(input_para, dim=1)
        else:
            xy = input_para
        return self.net(xy)

    
   





#-------------------------------------------------------------------------------------------#
# Geometric sampling section (same as before, corners belong to the walls)
# ----------------------------------------------------------------------------

N = 70     
# geometric sampling section
x = torch.linspace(0.0, 1.0, N)
y = torch.linspace(0.0, 1.0, N)
X,  Y = torch.meshgrid(x, y, indexing="ij")
xy_all = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
 
mask_b_c = ((Y == 0) | (Y == 1)).reshape(-1)
mask_inlet = (((X == 0) | (X == 1)) & (Y > 0) & (Y < 1)).reshape(-1)
 
xy_bnd_bc = xy_all[mask_b_c].to(device)                    # walls (y*=0,1)
xy_bnd_inlet = xy_all[mask_inlet].to(device)               # x*=0 and x*=1 edges
xy_int = xy_all[~(mask_b_c | mask_inlet)].to(device).requires_grad_(True)
 
# Split inlet / outlet
xy_in = xy_bnd_inlet[xy_bnd_inlet[:, 0] == 0.0]
xy_out = xy_bnd_inlet[xy_bnd_inlet[:, 0] == 1.0]
print("format:xy_int,xy_bnd_bc,xy_in,xy_out")


#-----------------------------------------------------------------------------
# LOSS EVALUATOR MODULES
# ----------------------------------------------------------------------------
"""
Non-dimensional Hagen-Poiseuille flow between plates (PINN loss section)
 
Scales:
    x* = x/x0,  y* = y/h
    u* = u/u0,  u0 = 1 (arbirtary)
    p* = p h^2 / (mu u0 x0)
 
Network: (x*, y*) -> (u*, p*)
 
Non-dimensional equations (v = 0, fully developed):
    continuity   : du*/dx*                      = 0
    x-momentum   : dp*/dx* - d2u*/dy*2          = 0
    y-momentum   : dp*/dy*                      = 0
 
"""
# ----------------------------------------------------------------------------
# Loss section
# ---------------------------------------------------------------------------- 
def grad(f, xy):        #<------------------------------------------ Diffential Operator
        """Gradient of scalar field f (N,1) w.r.t. xy (N,2) -> (N,2): [d/dx*, d/dy*]"""
        return torch.autograd.grad(f, xy, grad_outputs=torch.ones_like(f),
                                   create_graph=True)[0]

 
def Loss(model, xy, P_in=0, P_out=-2,
         W=dict(cont=1.0, momx=1.0, momy=1.0, constP=1, wall=10.0, pres=10.0)):
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
    res_constP = p_xx
    return (torch.mean(res_cont ** 2),
            torch.mean(res_momx ** 2),
            torch.mean(res_momy ** 2),
            torch.mean(res_constP ** 2))


def bc_loss(model, xy_bnd_bc, xy_in, xy_out, P_in=0, P_out=-2):
    # No-slip on the walls: u* = 0
    loss_wall = torch.mean(model(xy_bnd_bc)[:, 0:1] ** 2)
    # Pressure fixed at inlet and outlet (drives the flow)
    loss_pin = torch.mean((model(xy_in)[:, 1:2] - P_in) ** 2)
    loss_pout = torch.mean((model(xy_out)[:, 1:2] - P_out) ** 2)
    return loss_wall, loss_pin + loss_pout





#--------------------------------------------------------------------------------------#
# TRAINING LOOP                                                                        #
#--------------------------------------------------------------------------------------#
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

model = NavierStokesPINN_SIMPLE(in_features=2, num_layers=9, num_hidden=160, out_features=2,
                 activation_func=nn.SiLU,
                 init_type_weight="xavier_normal", init_type_bias="zeros").to(device)

# ---- settings ---------------------------------------------------------------
ADAM_LR = 2e-3          # initial learning rate
MIN_LR = 1e-6           # Adam stops decaying / hands over at this LR
LR_FACTOR = 0.5         # LR is multiplied by this on every plateau
LR_PATIENCE = 200       # epochs without improvement before decaying the LR
SWITCH_LOSS = 1e-4      # switch Adam -> L-BFGS once unweighted loss is below this
MAX_ADAM = 20000        # hard cap on Adam epochs
LBFGS_ROUNDS = 20       # max outer L-BFGS rounds
LBFGS_ITERS = 200       # inner iterations per round
TOL = 1e-10             # stop completely once unweighted loss is below this
LOG_EVERY = 200


# ---- losses -----------------------------------------------------------------
def compute_losses(model):
    """
    weighted   : used ONLY for backprop / optimisation (uses W)
    plain      : unweighted sum of all terms, used for monitoring, LR
                 scheduling, best-model selection and the Adam->L-BFGS switch
    parts      : individual unweighted terms
    """
    l_c, l_mx, l_my, l_cp = Loss(model, xy_int)
    l_wall, l_pres = bc_loss(model,xy_bnd_bc,xy_in,xy_out)
    weighted = (W["cont"] * l_c + W["momx"] * l_mx + W["momy"] * l_my
                + W["constP"] * l_cp + W["wall"] * l_wall + W["pres"] * l_pres)
    plain = l_c + l_mx + l_my + l_cp + l_wall + l_pres
    parts = dict(cont=l_c.item(), momx=l_mx.item(), momy=l_my.item(),
                 constP=l_cp.item(), wall=l_wall.item(), pres=l_pres.item())
    return weighted, plain, parts


history = []                       # unweighted loss history
best_loss, best_state = float("inf"), None


def log(tag, it, plain, parts, t0):
    print(f"[{tag} {it:6d}] loss={plain:.3e} | "
          + " ".join(f"{k}={v:.1e}" for k, v in parts.items())
          + f" | {time.time() - t0:.0f}s")


# ---- Stage 1: Adam with plateau-based LR reduction ---------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
    min_lr=MIN_LR)

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

    if ep % LOG_EVERY == 0:
        log("Adam", ep, plain_val, parts, t0)
        print(f"             lr={optimizer.param_groups[0]['lr']:.2e}")

    if plain_val < SWITCH_LOSS:
        print(f"\nUnweighted loss < {SWITCH_LOSS:.0e} at epoch {ep}: "
              f"switching to L-BFGS")
        break
    if optimizer.param_groups[0]["lr"] <= MIN_LR * 1.0001 and ep > LR_PATIENCE:
        print(f"\nLR reached minimum at epoch {ep}: switching to L-BFGS")
        break

# ---- Stage 2: L-BFGS in rounds, restarted from the best weights --------------
model.load_state_dict(best_state)
stall = 0
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

    _, plain, parts = compute_losses(model)
    plain_val = plain.item()
    history.append(plain_val)
    log("L-BFGS round", rnd, plain_val, parts, t0)

    if plain_val < best_loss:
        stall = stall + 1 if plain_val > 0.99 * best_loss else 0
        best_loss, best_state = plain_val, copy.deepcopy(model.state_dict())
    else:
        stall += 1
        model.load_state_dict(best_state)      # discard a worse round

    if best_loss < TOL or stall >= 3:
        break

model.load_state_dict(best_state)
_, plain, parts = compute_losses(model)
print(f"\nFinal unweighted loss = {plain.item():.3e}")
print("Terms:", " ".join(f"{k}={v:.2e}" for k, v in parts.items()))
training_time = time.time() - t0
# ---- Results (no reference solution used) ------------------------------------
model.eval()
with torch.no_grad():
    out = model(xy_all.to(device)).cpu().numpy()
n = int(np.sqrt(xy_all.shape[0]))
U = out[:, 0].reshape(n, n)
P = out[:, 1].reshape(n, n)

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
plt.savefig("pinn_results.png", dpi=150)

#--------------------------------------------------------------------------------#
# TRAINING SET FROM THE EXACT NON-DIMENSIONAL SOLUTION                           #
#--------------------------------------------------------------------------------#
def exact_solution(xy, P_in=0.0, P_out=-2.0):
    """
    Exact non-dimensional Hagen-Poiseuille solution on [0,1]x[0,1]
    (u0 = 1 arbitrary, so the flow strength is set by the pressure drop):
 
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
    """
    Draw n_data random points uniformly from [0,1]x[0,1] and evaluate the
    exact solution there.  Returns (xy_data, u_data, p_data) on `device`.
    The seed makes the point set reproducible.
    """
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n_data, 2, generator=g)            # random (x*, y*) points
    u, p = exact_solution(xy, P_in, P_out)
    return xy.to(device), u.to(device), p.to(device)
 
 
def data_loss(model, xy_data, u_data, p_data, eps=1e-20):
    """
    RMS error of the network against the training set, for u* and p*:
        rms_u = sqrt( mean( (u_pred - u_data)^2 ) )
    eps keeps the gradient finite if the error ever becomes exactly zero.
    """
    out = model(xy_data)
    rms_u = torch.sqrt(torch.mean((out[:, 0:1] - u_data) ** 2) + eps)
    rms_p = torch.sqrt(torch.mean((out[:, 1:2] - p_data) ** 2) + eps)
    return rms_u, rms_p


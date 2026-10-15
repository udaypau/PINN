
#this directory contains case specific pde loss function methods 
import torch
import torch.nn as nn
import pandas as pd


class LidDrivenCavity:
    def __init__(
        self,
        Re=100.0,
        U_lid=1.0,
        L=1.0,
        t_final=1.0,
        n_points_per_wall=250,
        n_ic_points=1000,
        n_pde_points=2500,
        weights=None,
        device="cpu"
    ):
        """
        Consolidated 2D unsteady Lid-Driven Cavity solver module with
        internal 4x3 State-Jacobian weight annealing.
        """
        self.device = device
        self.Re = float(Re)
        self.inv_Re = 1.0 / self.Re
        self.U_lid = float(U_lid)
        self.L = float(L)
        self.t_final = float(t_final)

        self.n_points_per_wall = n_points_per_wall
        self.n_ic_points = n_ic_points
        self.n_pde_points = n_pde_points
        self.t_rand = torch.empty(self.n_points_per_wall, 1, device=self.device).uniform_(0.0, self.t_final)
        self.s_rand = torch.empty(self.n_points_per_wall, 1, device=self.device).uniform_(0.0, self.L)
        self.x_ic = torch.empty(self.n_ic_points, 1, device=self.device).uniform_(0.0, self.L)
        self.y_ic = torch.empty(self.n_ic_points, 1, device=self.device).uniform_(0.0, self.L)
        self.t_ic = torch.zeros(self.n_ic_points, 1, device=self.device)
        
        self.weights = weights if weights is not None else {
            "pde": 1.0,
            "bc": 1.0,
            "ic": 1.0,
            "data": 1.0
        }

        # Annealing state trackers
        self.iteration_count = 0
        self.alpha_ema = 0.1  # EMA smoothing factor for weight updates

        # Summary printed once upon instantiation
        print(f"""
================================================================================
                    LID-DRIVEN CAVITY: SYSTEM CONFIGURATION
================================================================================
1. DATA SLICING SPECIFICATION:
   Expected sequence : [t, x, y, z, u, v, w, p]
   Retained fields   : [t (0), x (1), y (2), u (4), v (5), p (7)]
   Omitted fields    : z (Col 3), w (Col 6)

2. DOMAINS & PHYSICAL PARAMETERS:
   - Reynolds Number (Re) : {self.Re} (1/Re = {self.inv_Re:.5e})
   - Spatial Span         : x in [0.0, {self.L}], y in [0.0, {self.L}]
   - Time Span            : t in [0.0, {self.t_final}]
   - Lid Velocity (U_lid) : {self.U_lid} (top wall y = {self.L})
   - Collocation Points   : PDE={self.n_pde_points}, Wall={self.n_points_per_wall}/wall, IC={self.n_ic_points}

3. GOVERNING EQUATIONS (Non-Dimensional Incompressible Navier-Stokes):
   - Continuity : ∂u/∂x + ∂v/∂y = 0
   - x-Momentum : ∂u/∂t + u(∂u/∂x) + v(∂u/∂y) + ∂p/∂x - (1/Re)(∂²u/∂x² + ∂²u/∂y²) = 0
   - y-Momentum : ∂v/∂t + u(∂v/∂x) + v(∂v/∂y) + ∂p/∂y - (1/Re)(∂²v/∂x² + ∂²v/∂y²) = 0

4. BOUNDARY & INITIAL CONDITIONS:
   - Top Wall (y = L)    : u = {self.U_lid}, v = 0, ∂p/∂y = 0
   - Bottom Wall (y = 0) : u = 0, v = 0, ∂p/∂y = 0
   - Left Wall (x = 0)   : u = 0, v = 0, ∂p/∂x = 0
   - Right Wall (x = L)  : u = 0, v = 0, ∂p/∂x = 0
   - Initial State (t = 0): u = 0, v = 0, p = 0 across whole domain

5. LOSS WEIGHTS (Initial):
   {self.weights}
================================================================================
""")

    def LDC_solver(self, model, data_df):
        """
        Slices pandas data, computes autograd derivatives, enforces PDEs, BCs, ICs,
        and supervised data loss in a single execution.
        
        Parameters:
          model   : Instantiated NavierStokesPINN network.
          data_df : Pandas DataFrame with layout [t, x, y, z, u, v, w, p]
          
        Returns:
          total_loss (torch.Tensor): Weighted objective for backward().
          loss_dict (dict)         : Breakdown of individual loss values.
        """
        # =====================================================================
        # 1. PARSE & SLICE PANDAS DATA (OMIT z and w)
        # =====================================================================
        # Column mapping: 0->t, 1->x, 2->y, 4->u, 5->v, 7->p
        vals = data_df.values
        
        t_data = torch.tensor(vals[:, 0:1], dtype=torch.float32, device=self.device)
        x_data = torch.tensor(vals[:, 1:2], dtype=torch.float32, device=self.device)
        y_data = torch.tensor(vals[:, 2:3], dtype=torch.float32, device=self.device)
        u_data = torch.tensor(vals[:, 4:5], dtype=torch.float32, device=self.device)
        v_data = torch.tensor(vals[:, 5:6], dtype=torch.float32, device=self.device)
        p_data = torch.tensor(vals[:, 7:8], dtype=torch.float32, device=self.device)
        
    
        
        # Supervised data loss
        pred_data = model([x_data, y_data, t_data])
        loss_data = (
            torch.mean((pred_data[:, 0:1] - u_data) ** 2) +
            torch.mean((pred_data[:, 1:2] - v_data) ** 2) +
            torch.mean((pred_data[:, 2:3] - p_data) ** 2)
        )

        # =====================================================================
        # 2. PDE RESIDUAL LOSS (INTERIOR COLLOCATION + AUTOGRAD)
        # =====================================================================
        x_c = torch.empty(self.n_pde_points, 1, device=self.device).uniform_(0.0, self.L).requires_grad_(True)
        y_c = torch.empty(self.n_pde_points, 1, device=self.device).uniform_(0.0, self.L).requires_grad_(True)
        t_c = torch.empty(self.n_pde_points, 1, device=self.device).uniform_(0.0, self.t_final).requires_grad_(True)

        pred_pde = model([x_c, y_c, t_c])
        u = pred_pde[:, 0:1]
        v = pred_pde[:, 1:2]
        p = pred_pde[:, 2:3]

        # First derivatives
        u_x = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y_c, torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]

        v_x = torch.autograd.grad(v, x_c, torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y_c, torch.ones_like(v), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t_c, torch.ones_like(v), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x_c, torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y_c, torch.ones_like(p), create_graph=True)[0]

        # Second derivatives (viscous diffusion)
        u_xx = torch.autograd.grad(u_x, x_c, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_c, torch.ones_like(u_y), create_graph=True)[0]

        v_xx = torch.autograd.grad(v_x, x_c, torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y_c, torch.ones_like(v_y), create_graph=True)[0]

        # Residuals
        res_cont = u_x + v_y
        res_mom_x = u_t + (u * u_x + v * u_y) + p_x - self.inv_Re * (u_xx + u_yy)
        res_mom_y = v_t + (u * v_x + v * v_y) + p_y - self.inv_Re * (v_xx + v_yy)

        loss_pde = (
            torch.mean(res_cont ** 2) +
            torch.mean(res_mom_x ** 2) +
            torch.mean(res_mom_y ** 2)
        )

        # =====================================================================
        # 3. BOUNDARY CONDITIONS (WALLS + NORMAL PRESSURE GRADIENTS)
        # =====================================================================
       

        # Top Lid (y = L) -> u = U_lid, v = 0, dp/dy = 0
        x_top = self.s_rand.clone().requires_grad_(True)
        y_top = torch.full_like(x_top, self.L).requires_grad_(True)
        pred_top = model([x_top, y_top, self.t_rand])
        p_top = pred_top[:, 2:3]
        p_y_top = torch.autograd.grad(p_top, y_top, torch.ones_like(p_top), create_graph=True)[0]
        loss_top = (
            torch.mean((pred_top[:, 0:1] - self.U_lid) ** 2) +
            torch.mean(pred_top[:, 1:2] ** 2) +
            torch.mean(p_y_top ** 2)
        )

        # Bottom Wall (y = 0) -> u = 0, v = 0, dp/dy = 0
        x_bot = self.s_rand.clone().requires_grad_(True)
        y_bot = torch.zeros_like(x_bot).requires_grad_(True)
        pred_bot = model([x_bot, y_bot, self.t_rand])
        p_bot = pred_bot[:, 2:3]
        p_y_bot = torch.autograd.grad(p_bot, y_bot, torch.ones_like(p_bot), create_graph=True)[0]
        loss_bot = (
            torch.mean(pred_bot[:, 0:1] ** 2) +
            torch.mean(pred_bot[:, 1:2] ** 2) +
            torch.mean(p_y_bot ** 2)
        )

        # Left Wall (x = 0) -> u = 0, v = 0, dp/dx = 0
        x_left = torch.zeros_like(self.s_rand).requires_grad_(True)
        y_left = self.s_rand.clone().requires_grad_(True)
        pred_left = model([x_left, y_left, self.t_rand])
        p_left = pred_left[:, 2:3]
        p_x_left = torch.autograd.grad(p_left, x_left, torch.ones_like(p_left), create_graph=True)[0]
        loss_left = (
            torch.mean(pred_left[:, 0:1] ** 2) +
            torch.mean(pred_left[:, 1:2] ** 2) +
            torch.mean(p_x_left ** 2)
        )

        # Right Wall (x = L) -> u = 0, v = 0, dp/dx = 0
        x_right = torch.full_like(self.s_rand, self.L).requires_grad_(True)
        y_right = self.s_rand.clone().requires_grad_(True)
        pred_right = model([x_right, y_right, self.t_rand])
        p_right = pred_right[:, 2:3]
        p_x_right = torch.autograd.grad(p_right, x_right, torch.ones_like(p_right), create_graph=True)[0]
        loss_right = (
            torch.mean(pred_right[:, 0:1] ** 2) +
            torch.mean(pred_right[:, 1:2] ** 2) +
            torch.mean(p_x_right ** 2)
        )

        loss_bc = loss_top + loss_bot + loss_left + loss_right

        # =====================================================================
        # 4. INITIAL CONDITION (t = 0: u = 0, v = 0, p = 0)
        # =====================================================================
        x_ic = torch.empty(self.n_ic_points, 1, device=self.device).uniform_(0.0, self.L)
        y_ic = torch.empty(self.n_ic_points, 1, device=self.device).uniform_(0.0, self.L)
        t_ic = torch.zeros(self.n_ic_points, 1, device=self.device)

        pred_ic = model([x_ic, y_ic, t_ic])
        loss_ic = (
            torch.mean(pred_ic[:, 0:1] ** 2) +
            torch.mean(pred_ic[:, 1:2] ** 2) +
            torch.mean(pred_ic[:, 2:3] ** 2)
        )

        # =====================================================================
        # 5. TOTAL WEIGHTED LOSS
        # =====================================================================
        total_loss = (
            self.weights["pde"] * loss_pde +
            self.weights["bc"] * loss_bc +
            self.weights["ic"] * loss_ic +
            self.weights["data"] * loss_data
        )

        loss_dict = {
            "total": total_loss,
            "pde": loss_pde,
            "bc": loss_bc,
            "ic": loss_ic,
            "data": loss_data
        }

        return total_loss, loss_dict 



    def LDC_solver_weightannealing(self, model, data_df, eps=1e-5):
        """
        Executes PDE, BC, IC, and data loss evaluation directly. 
        Every 25 calls, an isolated finite-perturbation routine constructs 
        the 4x3 Loss-to-State Jacobian matrix to update weights.
        """
        # ---------------------------------------------------------------------
        # 1. PARSE & SLICE DATA (Omit z and w)
        # ---------------------------------------------------------------------
        vals = data_df.values
        t_data = torch.tensor(vals[:, 0:1], dtype=torch.float32, device=self.device)
        x_data = torch.tensor(vals[:, 1:2], dtype=torch.float32, device=self.device)
        y_data = torch.tensor(vals[:, 2:3], dtype=torch.float32, device=self.device)
        u_data = torch.tensor(vals[:, 4:5], dtype=torch.float32, device=self.device)
        v_data = torch.tensor(vals[:, 5:6], dtype=torch.float32, device=self.device)
        p_data = torch.tensor(vals[:, 7:8], dtype=torch.float32, device=self.device)

        # ---------------------------------------------------------------------
        # 2. CORE LOSS EVALUATION (No Wrappers, No Phantom Variables)
        # ---------------------------------------------------------------------
        def evaluate_losses(shift=None):
            # --- Supervised Data Loss ---
            p_dat = model([x_data, y_data, t_data])
            if shift is not None:
                p_dat = p_dat + shift
            l_data = (
                torch.mean((p_dat[:, 0:1] - u_data) ** 2) +
                torch.mean((p_dat[:, 1:2] - v_data) ** 2) +
                torch.mean((p_dat[:, 2:3] - p_data) ** 2)
            )

            # --- PDE Residual Loss ---
            x_c = torch.empty(self.n_pde_points, 1, device=self.device).uniform_(0.0, self.L).requires_grad_(True)
            y_c = torch.empty(self.n_pde_points, 1, device=self.device).uniform_(0.0, self.L).requires_grad_(True)
            t_c = torch.empty(self.n_pde_points, 1, device=self.device).uniform_(0.0, self.t_final).requires_grad_(True)

            p_pde = model([x_c, y_c, t_c])
            if shift is not None:
                p_pde = p_pde + shift

            u = p_pde[:, 0:1]
            v = p_pde[:, 1:2]
            p = p_pde[:, 2:3]

            u_x = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
            u_y = torch.autograd.grad(u, y_c, torch.ones_like(u), create_graph=True)[0]
            u_t = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]

            v_x = torch.autograd.grad(v, x_c, torch.ones_like(v), create_graph=True)[0]
            v_y = torch.autograd.grad(v, y_c, torch.ones_like(v), create_graph=True)[0]
            v_t = torch.autograd.grad(v, t_c, torch.ones_like(v), create_graph=True)[0]

            p_x = torch.autograd.grad(p, x_c, torch.ones_like(p), create_graph=True)[0]
            p_y = torch.autograd.grad(p, y_c, torch.ones_like(p), create_graph=True)[0]

            u_xx = torch.autograd.grad(u_x, x_c, torch.ones_like(u_x), create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, y_c, torch.ones_like(u_y), create_graph=True)[0]

            v_xx = torch.autograd.grad(v_x, x_c, torch.ones_like(v_x), create_graph=True)[0]
            v_yy = torch.autograd.grad(v_y, y_c, torch.ones_like(v_y), create_graph=True)[0]

            res_cont = u_x + v_y
            res_mom_x = u_t + (u * u_x + v * u_y) + p_x - self.inv_Re * (u_xx + u_yy)
            res_mom_y = v_t + (u * v_x + v * v_y) + p_y - self.inv_Re * (v_xx + v_yy)

            l_pde = torch.mean(res_cont ** 2) + torch.mean(res_mom_x ** 2) + torch.mean(res_mom_y ** 2)

            # --- Boundary Conditions (Dirichlet + dp/dn = 0) ---
            # Top Wall (y = L) -> dp/dy = 0
            x_top = self.s_rand.clone().requires_grad_(True)
            y_top = torch.full_like(x_top, self.L).requires_grad_(True)
            p_top = model([x_top, y_top, self.t_rand])
            if shift is not None:
                p_top = p_top + shift
            p_val_top = p_top[:, 2:3]
            p_y_top = torch.autograd.grad(p_val_top, y_top, torch.ones_like(p_val_top), create_graph=True)[0]
            l_top = (
                torch.mean((p_top[:, 0:1] - self.U_lid) ** 2) +
                torch.mean(p_top[:, 1:2] ** 2) +
                torch.mean(p_y_top ** 2)
            )

            # Bottom Wall (y = 0) -> dp/dy = 0
            x_bot = self.s_rand.clone().requires_grad_(True)
            y_bot = torch.zeros_like(x_bot).requires_grad_(True)
            p_bot = model([x_bot, y_bot, self.t_rand])
            if shift is not None:
                p_bot = p_bot + shift
            p_val_bot = p_bot[:, 2:3]
            p_y_bot = torch.autograd.grad(p_val_bot, y_bot, torch.ones_like(p_val_bot), create_graph=True)[0]
            l_bot = (
                torch.mean(p_bot[:, 0:1] ** 2) +
                torch.mean(p_bot[:, 1:2] ** 2) +
                torch.mean(p_y_bot ** 2)
            )

            # Left Wall (x = 0) -> dp/dx = 0
            x_left = torch.zeros_like(self.s_rand).requires_grad_(True)
            y_left = self.s_rand.clone().requires_grad_(True)
            p_left = model([x_left, y_left, self.t_rand])
            if shift is not None:
                p_left = p_left + shift
            p_val_left = p_left[:, 2:3]
            p_x_left = torch.autograd.grad(p_val_left, x_left, torch.ones_like(p_val_left), create_graph=True)[0]
            l_left = (
                torch.mean(p_left[:, 0:1] ** 2) +
                torch.mean(p_left[:, 1:2] ** 2) +
                torch.mean(p_x_left ** 2)
            )

            # Right Wall (x = L) -> dp/dx = 0
            x_right = torch.full_like(self.s_rand, self.L).requires_grad_(True)
            y_right = self.s_rand.clone().requires_grad_(True)
            p_right = model([x_right, y_right, self.t_rand])
            if shift is not None:
                p_right = p_right + shift
            p_val_right = p_right[:, 2:3]
            p_x_right = torch.autograd.grad(p_val_right, x_right, torch.ones_like(p_val_right), create_graph=True)[0]
            l_right = (
                torch.mean(p_right[:, 0:1] ** 2) +
                torch.mean(p_right[:, 1:2] ** 2) +
                torch.mean(p_x_right ** 2)
            )

            l_bc = l_top + l_bot + l_left + l_right

            # --- Initial Conditions ---
            if not hasattr(self, 'x_ic'):
                self.x_ic = torch.empty(self.n_ic_points, 1, device=self.device).uniform_(0.0, self.L)
                self.y_ic = torch.empty(self.n_ic_points, 1, device=self.device).uniform_(0.0, self.L)
                self.t_ic = torch.zeros(self.n_ic_points, 1, device=self.device)

            p_ic = model([self.x_ic, self.y_ic, self.t_ic])
            if shift is not None:
                p_ic = p_ic + shift
            l_ic = (
                torch.mean(p_ic[:, 0:1] ** 2) +
                torch.mean(p_ic[:, 1:2] ** 2) +
                torch.mean(p_ic[:, 2:3] ** 2)
            )

            return [l_pde, l_bc, l_ic, l_data]

        # ---------------------------------------------------------------------
        # 3. BASELINE EVALUATION (Zero Shifts)
        # ---------------------------------------------------------------------
        base_losses = evaluate_losses(shift=None)
        loss_pde, loss_bc, loss_ic, loss_data = base_losses
        loss_names = ["pde", "bc", "ic", "data"]

        # ---------------------------------------------------------------------
        # 4. ISOLATED 4x3 FINITE-PERTURBATION JACOBIAN (Only Every 25 Epochs)
        # ---------------------------------------------------------------------
        if self.iteration_count > 0 and self.iteration_count % 25 == 0:
            with torch.no_grad():
                jacobian = torch.zeros(4, 3, device=self.device)
                for col in range(3):
                    shift_vec = torch.zeros(1, 3, device=self.device)
                    shift_vec[0, col] = eps
                    
                    pert_losses = evaluate_losses(shift=shift_vec)
                    for row in range(4):
                        diff = torch.abs(pert_losses[row] - base_losses[row])
                        jacobian[row, col] = diff / eps

                row_sens = torch.sum(jacobian, dim=1) + 1e-8
                mean_sens = torch.mean(row_sens)
                target_weights = mean_sens / row_sens

                for idx, key in enumerate(loss_names):
                    self.weights[key] = (
                        (1.0 - self.alpha_ema) * self.weights[key] +
                        self.alpha_ema * target_weights[idx].item()
                    )

                print(f"\n[Epoch {self.iteration_count} Annealing] 4x3 Jacobian Matrix (|dL_i/d(u,v,p)|):")
                for idx, name in enumerate(loss_names):
                    print(f"  {name:4s} -> du: {jacobian[idx, 0]:.3e} | dv: {jacobian[idx, 1]:.3e} | dp: {jacobian[idx, 2]:.3e} | Row Sum: {row_sens[idx].item():.3e}")
                print(f"  Updated Weights: { {k: round(v, 4) for k, v in self.weights.items()} }\n")

        self.iteration_count += 1

        # ---------------------------------------------------------------------
        # 5. TOTAL OBJECTIVE FOR BACKWARD
        # ---------------------------------------------------------------------
        total_loss = (
            self.weights["pde"] * loss_pde +
            self.weights["bc"] * loss_bc +
            self.weights["ic"] * loss_ic +
            self.weights["data"] * loss_data
        )

        loss_dict = {
            "total": total_loss,
            "pde": loss_pde,
            "bc": loss_bc,
            "ic": loss_ic,
            "data": loss_data,
            "current_weights": self.weights.copy()
        }

        return total_loss, loss_dict
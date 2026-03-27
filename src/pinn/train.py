"""Training loops for forward and inverse PINN beam solvers.

Key insight: PINNs fail without non-dimensionalization. The Euler-Bernoulli
equation mixes quantities spanning 10+ orders of magnitude (E~210e9, w~7e-4).

Non-dimensional formulation:
    x̄ = x/L ∈ [0,1]
    w̄ = w * EI/(qL⁴)
    PDE: d⁴w̄/dx̄⁴ = 1  (all terms O(1))

Training recipe: Adam warmup → L-BFGS refinement (standard for PINNs).
"""

import torch
import torch.nn as nn
import numpy as np
from .model import BeamPINN, compute_derivatives


# ── Forward Solver (non-dimensional) ──────────────────────────

def forward_loss(model, x_bar, lambda_bc=100.0):
    """Non-dimensional PDE loss: d⁴w̄/dx̄⁴ = 1, with clamped-free BCs."""
    derivs = compute_derivatives(model, x_bar)
    loss_pde = torch.mean((derivs["d4w_dx4"] - 1.0) ** 2)

    # BCs at x̄=0 (clamped)
    x0 = torch.zeros(1, 1, requires_grad=True)
    d0 = compute_derivatives(model, x0)

    # BCs at x̄=1 (free)
    x1 = torch.ones(1, 1, requires_grad=True)
    d1 = compute_derivatives(model, x1)

    loss_bc = (d0["w"] ** 2 + d0["dw_dx"] ** 2
               + d1["d2w_dx2"] ** 2 + d1["d3w_dx3"] ** 2).squeeze()

    total = loss_pde + lambda_bc * loss_bc
    return total, loss_pde.item(), loss_bc.item()


def train_forward(beam_params, n_colloc=300, n_adam=5000, n_lbfgs=2000,
                  lr=1e-3, lambda_bc=100.0,
                  hidden_dims=(64, 64, 64, 64), print_every=1000):
    """Train forward PINN with Adam warmup + L-BFGS refinement."""
    model = BeamPINN(hidden_dims)
    x_bar = torch.linspace(0.01, 0.99, n_colloc).unsqueeze(1)

    losses = {"total": [], "pde": [], "bc": []}

    def record(loss, pde, bc):
        losses["total"].append(loss)
        losses["pde"].append(pde)
        losses["bc"].append(bc)

    # Phase 1: Adam warmup
    print("    Phase 1: Adam warmup...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_adam, eta_min=1e-5)

    for epoch in range(1, n_adam + 1):
        optimizer.zero_grad()
        loss, pde, bc = forward_loss(model, x_bar, lambda_bc)
        loss.backward()
        optimizer.step()
        scheduler.step()
        record(loss.item(), pde, bc)

        if epoch % print_every == 0 or epoch == 1:
            print(f"    Epoch {epoch:5d}/{n_adam}: loss={loss.item():.2e} "
                  f"pde={pde:.2e} bc={bc:.2e}")

    # Phase 2: L-BFGS refinement
    print("    Phase 2: L-BFGS refinement...")
    optimizer2 = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )

    _last = {}
    for step in range(1, n_lbfgs + 1):
        def closure():
            optimizer2.zero_grad()
            l, p, b = forward_loss(model, x_bar, lambda_bc)
            l.backward()
            _last.update(loss=l.item(), pde=p, bc=b)
            return l

        optimizer2.step(closure)
        record(_last["loss"], _last["pde"], _last["bc"])

        if step % (n_lbfgs // 5) == 0 or step == 1:
            print(f"    L-BFGS {step:4d}/{n_lbfgs}: loss={_last['loss']:.2e} "
                  f"pde={_last['pde']:.2e} bc={_last['bc']:.2e}")

    return model, losses


# ── Inverse Solver ─────────────────────────────────────────────

class InverseBeamPINN(nn.Module):
    """PINN that learns w̄(x̄) and discovers unknown E simultaneously.

    Non-dim PDE: d⁴w̄/dx̄⁴ = E_ref / E  (ratio → O(1) target)
    Data loss: w̄(x̄_i) vs w̄_data_i (pre-scaled to non-dim)
    """

    def __init__(self, E_init, hidden_dims=(64, 64, 64, 64)):
        super().__init__()
        self.beam_net = BeamPINN(hidden_dims)
        # Parameterize in log-space for positivity
        self.log_E = nn.Parameter(torch.tensor(np.log(E_init), dtype=torch.float32))

    @property
    def E(self):
        return torch.exp(self.log_E)

    def forward(self, x):
        return self.beam_net(x)


def inverse_loss(model, x_bar, x_data_bar, w_data_bar, E_ref,
                 lambda_bc=100.0, lambda_data=200.0):
    """Inverse problem loss in non-dimensional space."""
    E = model.E
    rhs = E_ref / E  # non-dim PDE right-hand side

    # PDE residual
    derivs = compute_derivatives(model, x_bar)
    loss_pde = torch.mean((derivs["d4w_dx4"] - rhs) ** 2)

    # BCs
    x0 = torch.zeros(1, 1, requires_grad=True)
    d0 = compute_derivatives(model, x0)
    x1 = torch.ones(1, 1, requires_grad=True)
    d1 = compute_derivatives(model, x1)
    loss_bc = (d0["w"] ** 2 + d0["dw_dx"] ** 2
               + d1["d2w_dx2"] ** 2 + d1["d3w_dx3"] ** 2).squeeze()

    # Data loss (non-dimensional)
    w_pred = model(x_data_bar)
    loss_data = torch.mean((w_pred - w_data_bar) ** 2)

    total = loss_pde + lambda_bc * loss_bc + lambda_data * loss_data
    return total, loss_pde.item(), loss_bc.item(), loss_data.item(), E.item()


def train_inverse(beam_params, x_meas_dim, w_meas_dim,
                  E_init, E_ref, n_colloc=300,
                  n_adam=8000, n_lbfgs=3000, lr=1e-3,
                  lambda_bc=100.0, lambda_data=200.0,
                  hidden_dims=(64, 64, 64, 64), print_every=2000):
    """Train inverse PINN to discover E from measured deflections.

    Args:
        beam_params: dict with L, I, q (E is unknown)
        x_meas_dim: measurement positions (m)
        w_meas_dim: measured deflections (m)
        E_init: initial guess for E (Pa)
        E_ref: reference E for non-dimensionalization (Pa)
    """
    L = beam_params["L"]
    I = beam_params["I"]
    q = beam_params["q"]
    E_true = beam_params.get("E_true", None)

    # Non-dimensionalize measurements
    w_ref = q * L ** 4 / (E_ref * I)
    x_data_bar = (x_meas_dim / L).unsqueeze(1)
    w_data_bar = (w_meas_dim / w_ref).unsqueeze(1)

    model = InverseBeamPINN(E_init=E_init, hidden_dims=hidden_dims)
    x_bar = torch.linspace(0.01, 0.99, n_colloc).unsqueeze(1)

    losses = {"total": [], "pde": [], "bc": [], "data": []}
    E_history = []

    def record(t, p, b, d, e):
        losses["total"].append(t)
        losses["pde"].append(p)
        losses["bc"].append(b)
        losses["data"].append(d)
        E_history.append(e)

    # Phase 1: Adam
    print("    Phase 1: Adam warmup...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_adam, eta_min=1e-5)

    for epoch in range(1, n_adam + 1):
        optimizer.zero_grad()
        loss, p, b, d, e = inverse_loss(
            model, x_bar, x_data_bar, w_data_bar, E_ref, lambda_bc, lambda_data
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        record(loss.item(), p, b, d, e)

        if epoch % print_every == 0 or epoch == 1:
            true_str = f" (true={E_true/1e9:.0f})" if E_true else ""
            print(f"    Epoch {epoch:5d}/{n_adam}: loss={loss.item():.2e} "
                  f"E={e/1e9:.2f}GPa{true_str}")

    # Phase 2: L-BFGS
    print("    Phase 2: L-BFGS refinement...")
    optimizer2 = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )

    _last = {}
    for step in range(1, n_lbfgs + 1):
        def closure():
            optimizer2.zero_grad()
            l, p, b, d, e = inverse_loss(
                model, x_bar, x_data_bar, w_data_bar, E_ref, lambda_bc, lambda_data
            )
            l.backward()
            _last.update(loss=l.item(), pde=p, bc=b, data=d, E=e)
            return l

        optimizer2.step(closure)
        record(_last["loss"], _last["pde"], _last["bc"], _last["data"], _last["E"])

        if step % (n_lbfgs // 5) == 0 or step == 1:
            true_str = f" (true={E_true/1e9:.0f})" if E_true else ""
            print(f"    L-BFGS {step:4d}/{n_lbfgs}: loss={_last['loss']:.2e} "
                  f"E={_last['E']/1e9:.2f}GPa{true_str}")

    return model, losses, E_history

#!/usr/bin/env python3
"""Cantilever Beam PINN Solver — Training Pipeline

Solves the Euler-Bernoulli beam equation with physics-informed neural networks.
No training data — the network learns purely from the PDE and boundary conditions.

Two problems:
    1. Forward: Given loads + material → predict deflection w(x)
    2. Inverse: Given measured deflections → discover Young's modulus E

The PDE is non-dimensionalized so all quantities are O(1):
    x̄ = x/L,  w̄ = w·EI/(qL⁴),  PDE: d⁴w̄/dx̄⁴ = 1

Usage:
    cd beam-pinn-solver
    python train.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analytical.beam import (
    uniform_load, uniform_load_moment, uniform_load_shear,
    max_deflection_uniform,
)
from src.pinn.train import train_forward, train_inverse
from src.pinn.model import compute_derivatives

OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"

# ── Beam parameters (steel I-beam, 1m cantilever) ─────────────
BEAM = {
    "L": 1.0,          # Length (m)
    "E": 210e9,         # Young's modulus (Pa) — steel
    "I": 8.33e-6,       # Second moment of area (m⁴)
    "q": 10000.0,       # Uniform distributed load (N/m)
}


def evaluate_forward(model, beam):
    """Compare non-dim PINN solution to analytical, returning dimensional results."""
    L, E, I, q = beam["L"], beam["E"], beam["I"], beam["q"]
    w_scale = q * L**4 / (E * I)  # non-dim → dim conversion

    x_bar = torch.linspace(0, 1, 500).unsqueeze(1)
    x_dim = x_bar.numpy().flatten() * L

    model.eval()
    derivs = compute_derivatives(model, x_bar)
    w_bar = derivs["w"].detach().numpy().flatten()
    w_pinn = w_bar * w_scale  # convert to dimensional

    # Moment: M = EI * d²w/dx² = EI * (w_scale/L²) * d²w̄/dx̄²
    d2w_bar = derivs["d2w_dx2"].detach().numpy().flatten()
    m_pinn = E * I * (w_scale / L**2) * d2w_bar

    # Shear: V = -EI * d³w/dx³
    d3w_bar = derivs["d3w_dx3"].detach().numpy().flatten()
    v_pinn = -E * I * (w_scale / L**3) * d3w_bar

    w_exact = uniform_load(x_dim, L, q, E, I)
    m_exact = uniform_load_moment(x_dim, L, q, E, I)
    v_exact = uniform_load_shear(x_dim, L, q, E, I)
    w_max_exact = max_deflection_uniform(L, q, E, I)

    rel_err_w = np.abs(w_pinn - w_exact) / (np.abs(w_exact).max() + 1e-30)

    metrics = {
        "w_max_exact_mm": w_max_exact * 1000,
        "w_max_pinn_mm": float(w_pinn[-1]) * 1000,
        "w_max_error_pct": abs(float(w_pinn[-1]) - w_max_exact) / abs(w_max_exact) * 100,
        "w_mean_rel_error_pct": float(np.mean(rel_err_w[10:])) * 100,
    }

    return {
        "x_mm": x_dim * 1000, "x_dim": x_dim,
        "w_pinn": w_pinn, "w_exact": w_exact,
        "m_pinn": m_pinn, "m_exact": m_exact,
        "v_pinn": v_pinn, "v_exact": v_exact,
        "metrics": metrics,
    }


def plot_forward(result, losses, beam):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = result["x_mm"]

    # Deflection
    ax = axes[0, 0]
    ax.plot(x, result["w_exact"] * 1000, "k-", lw=2, label="Analytical")
    ax.plot(x, result["w_pinn"] * 1000, "r--", lw=2, label="PINN")
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Deflection (mm)")
    ax.set_title("Beam Deflection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # Bending moment
    ax = axes[0, 1]
    ax.plot(x, result["m_exact"], "k-", lw=2, label="Analytical")
    ax.plot(x, result["m_pinn"], "r--", lw=2, label="PINN")
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Bending Moment (N·m)")
    ax.set_title("Bending Moment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error
    ax = axes[1, 0]
    rel_err = np.abs(result["w_pinn"] - result["w_exact"]) / (np.abs(result["w_exact"]).max() + 1e-30) * 100
    ax.semilogy(x[10:], rel_err[10:], "b-", lw=1.5)
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Deflection Error Along Beam")
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1, 1]
    n = len(losses["total"])
    ax.semilogy(range(1, n + 1), losses["total"], label="Total", lw=1)
    ax.semilogy(range(1, n + 1), losses["pde"], label="PDE", lw=1, alpha=0.7)
    ax.semilogy(range(1, n + 1), losses["bc"], label="BC", lw=1, alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (Adam → L-BFGS)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    m = result["metrics"]
    fig.suptitle(
        f"Cantilever Beam PINN — Forward Problem\n"
        f"L={beam['L']*1000:.0f}mm, q={beam['q']/1000:.0f}kN/m, E={beam['E']/1e9:.0f}GPa | "
        f"Tip: exact={m['w_max_exact_mm']:.4f}mm, PINN={m['w_max_pinn_mm']:.4f}mm "
        f"({m['w_max_error_pct']:.3f}% error)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIGURES / "forward_solution.png", dpi=150)
    plt.close()


def plot_inverse(E_history, losses, E_true, E_init):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    n = len(E_history)
    iters = range(1, n + 1)

    ax = axes[0]
    ax.plot(iters, [e / 1e9 for e in E_history], "b-", lw=1.5, label="Estimated E")
    ax.axhline(E_true / 1e9, color="k", ls="--", lw=1, label=f"True = {E_true/1e9:.0f} GPa")
    ax.axhline(E_init / 1e9, color="gray", ls=":", lw=1, label=f"Guess = {E_init/1e9:.0f} GPa")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("E (GPa)")
    ax.set_title("Young's Modulus Convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(iters, losses["total"], label="Total", lw=1)
    ax.semilogy(iters, losses["pde"], label="PDE", lw=1, alpha=0.7)
    ax.semilogy(iters, losses["data"], label="Data", lw=1, alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Inverse Training Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    E_err = [abs(e - E_true) / E_true * 100 for e in E_history]
    ax.semilogy(iters, E_err, "r-", lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("E Error (%)")
    ax.set_title("Parameter Discovery Accuracy")
    ax.grid(True, alpha=0.3)

    final_E = E_history[-1]
    final_err = abs(final_E - E_true) / E_true * 100
    fig.suptitle(
        f"Cantilever Beam PINN — Inverse Problem\n"
        f"True E={E_true/1e9:.0f}GPa, Guess={E_init/1e9:.0f}GPa → "
        f"Found={final_E/1e9:.2f}GPa ({final_err:.2f}% error)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIGURES / "inverse_solution.png", dpi=150)
    plt.close()

    return {"E_true_GPa": E_true / 1e9, "E_found_GPa": final_E / 1e9,
            "E_error_pct": final_err, "E_init_GPa": E_init / 1e9}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    w_max = max_deflection_uniform(BEAM["L"], BEAM["q"], BEAM["E"], BEAM["I"])

    print("=" * 60)
    print("Cantilever Beam PINN Solver")
    print("=" * 60)
    print(f"  Beam: L={BEAM['L']*1000:.0f}mm, E={BEAM['E']/1e9:.0f}GPa, "
          f"I={BEAM['I']:.2e}m\u2074")
    print(f"  Load: q={BEAM['q']/1000:.0f} kN/m (uniform)")
    print(f"  Analytical tip deflection: {w_max*1000:.4f} mm")
    print(f"  Non-dim formulation: x\u0304=x/L, w\u0304=w\u00b7EI/(qL\u2074), PDE: d\u2074w\u0304/dx\u0304\u2074=1")

    # ── Forward Problem ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[1/2] Forward Problem: Predict deflection from PDE + BCs only")
    print(f"{'='*60}")

    fwd_model, fwd_losses = train_forward(
        BEAM, n_colloc=300, n_adam=5000, n_lbfgs=2000,
        lr=1e-3, lambda_bc=100.0,
        hidden_dims=(64, 64, 64, 64), print_every=1000,
    )

    result = evaluate_forward(fwd_model, BEAM)
    m = result["metrics"]
    print(f"\n  Forward Results:")
    print(f"    Tip deflection — exact: {m['w_max_exact_mm']:.4f} mm, "
          f"PINN: {m['w_max_pinn_mm']:.4f} mm")
    print(f"    Tip error: {m['w_max_error_pct']:.4f}%")
    print(f"    Mean error: {m['w_mean_rel_error_pct']:.4f}%")

    plot_forward(result, fwd_losses, BEAM)
    torch.save(fwd_model.state_dict(), MODELS / "forward_pinn.pt")

    # ── Inverse Problem ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[2/2] Inverse Problem: Discover E from 10 noisy measurements")
    print(f"{'='*60}")

    E_true = BEAM["E"]
    E_init = 50e9  # Deliberately wrong: 50 GPa instead of 210 GPa
    E_ref = E_init  # Use initial guess as non-dim reference

    # Generate "sensor" data with 1% noise
    np.random.seed(42)
    n_sensors = 10
    x_meas_np = np.linspace(0.1 * BEAM["L"], BEAM["L"], n_sensors)
    w_meas_np = uniform_load(x_meas_np, BEAM["L"], BEAM["q"], E_true, BEAM["I"])
    noise = 0.01 * np.abs(w_meas_np).max() * np.random.randn(n_sensors)
    w_meas_np += noise

    x_meas = torch.tensor(x_meas_np, dtype=torch.float32)
    w_meas = torch.tensor(w_meas_np, dtype=torch.float32)

    inv_params = {**BEAM, "E_true": E_true}
    print(f"  {n_sensors} sensors along beam, 1% Gaussian noise")
    print(f"  Initial guess: E = {E_init/1e9:.0f} GPa (true = {E_true/1e9:.0f} GPa)")

    inv_model, inv_losses, E_hist = train_inverse(
        inv_params, x_meas, w_meas,
        E_init=E_init, E_ref=E_ref,
        n_colloc=300, n_adam=8000, n_lbfgs=3000,
        lr=1e-3, lambda_bc=100.0, lambda_data=200.0,
        hidden_dims=(64, 64, 64, 64), print_every=2000,
    )

    inv_metrics = plot_inverse(E_hist, inv_losses, E_true, E_init)
    torch.save(inv_model.state_dict(), MODELS / "inverse_pinn.pt")

    print(f"\n  Inverse Results:")
    print(f"    E discovered: {inv_metrics['E_found_GPa']:.2f} GPa "
          f"(true: {inv_metrics['E_true_GPa']:.0f} GPa)")
    print(f"    Error: {inv_metrics['E_error_pct']:.2f}%")

    # ── Summary ────────────────────────────────────────────────
    summary = {
        "beam": {k: v for k, v in BEAM.items()},
        "forward": result["metrics"],
        "inverse": inv_metrics,
    }
    with open(MODELS / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved:")
    for p in [FIGURES / "forward_solution.png", FIGURES / "inverse_solution.png",
              MODELS / "results.json"]:
        print(f"    {p}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()

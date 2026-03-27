"""Physics-Informed Neural Network for Euler-Bernoulli beam.

The PINN learns to satisfy:
    1. The governing PDE: EI * d⁴w/dx⁴ = q(x)
    2. Boundary conditions at x=0 (clamped) and x=L (free)

No training data is needed — the network learns purely from the physics.
"""

import torch
import torch.nn as nn


class BeamPINN(nn.Module):
    """Neural network that approximates beam deflection w(x).

    Architecture: x → [hidden layers with Tanh] → w(x)
    Tanh activation is preferred for PINNs because it's infinitely
    differentiable (we need 4th derivatives).
    """

    def __init__(self, hidden_dims=(64, 64, 64, 64), output_dim=1):
        super().__init__()
        layers = []
        prev = 1  # single input: normalized position x/L
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def compute_derivatives(model, x):
    """Compute w and its first 4 derivatives via autograd.

    Returns dict with w, dw_dx, d2w_dx2, d3w_dx3, d4w_dx4.
    """
    x = x.clone().detach().requires_grad_(True)
    w = model(x)

    grads = {}
    grads["w"] = w
    current = w
    for i, key in enumerate(["dw_dx", "d2w_dx2", "d3w_dx3", "d4w_dx4"], 1):
        current = torch.autograd.grad(
            current, x,
            grad_outputs=torch.ones_like(current),
            create_graph=True,
        )[0]
        grads[key] = current

    return grads

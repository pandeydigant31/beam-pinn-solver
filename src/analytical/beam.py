"""Analytical solutions for Euler-Bernoulli cantilever beam.

Closed-form solutions for common load cases, used as ground truth
to validate the PINN solver.

Coordinate system:
    x = 0 at fixed (clamped) end, x = L at free end
    w(x) = transverse deflection (positive downward)
"""

import numpy as np


def uniform_load(x, L, q, E, I):
    """Deflection under uniform distributed load q (N/m).

    Analytical: w(x) = q/(24EI) * (x⁴ - 4Lx³ + 6L²x²)
    """
    return q / (24 * E * I) * (x**4 - 4 * L * x**3 + 6 * L**2 * x**2)


def uniform_load_moment(x, L, q, E, I):
    """Bending moment M(x) = EI * w''(x) under uniform load.

    M(x) = q/2 * (L - x)²
    """
    return q / 2 * (L - x) ** 2


def uniform_load_shear(x, L, q, E, I):
    """Shear force V(x) = -EI * w'''(x) under uniform load.

    V(x) = q * (L - x)
    """
    return q * (L - x)


def point_load(x, L, P, E, I):
    """Deflection under point load P at the free end (x = L).

    Analytical: w(x) = P/(6EI) * (3Lx² - x³)
    """
    return P / (6 * E * I) * (3 * L * x**2 - x**3)


def point_load_moment(x, L, P, E, I):
    """Bending moment under tip point load.

    M(x) = P * (L - x)
    """
    return P * (L - x)


def max_deflection_uniform(L, q, E, I):
    """Maximum deflection at free end under uniform load.

    w_max = qL⁴ / (8EI)
    """
    return q * L**4 / (8 * E * I)


def max_deflection_point(L, P, E, I):
    """Maximum deflection at free end under point load.

    w_max = PL³ / (3EI)
    """
    return P * L**3 / (3 * E * I)

"""Spiral generation package."""

from .ode import generate_tight_spiral_ode
from .simple_spiral import generate_spiral_on_surface

__all__ = ["generate_spiral_on_surface", "generate_tight_spiral_ode"]

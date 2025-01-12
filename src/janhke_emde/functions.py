import numpy as np
from scipy.special import gamma


def gamma2d(x, y):
    """Compute the absolute value of the gamma function for complex input x + yi."""
    return np.abs(gamma(x + y * 1j))


def diff(f, x, y, dx, dy, step=1e-6):
    """Compute numerical derivative of f at (x,y) in direction (dx,dy)."""
    return (f(x + dx * step, y + dy * step) - f(x, y)) / step

import numpy as np


def unique_tolerance(a: np.ndarray, tol: float):
    ai = (a / tol).astype(int)
    _, idx = np.unique(ai, return_index=True, axis=0)
    return a[idx]


def diff(f, x, y, dx, dy, step=1e-6):
    """Compute numerical derivative of f at (x,y) in direction (dx,dy) using central finite difference."""
    return (f(x + dx * step, y + dy * step) - f(x - dx * step, y - dy * step)) / (
        2 * step
    )


def hessian(f, x, y, step=1e-6):
    """Compute the hessian matrix of f at (x,y) using central finite difference."""
    h = np.zeros(list(np.shape(x)) + [2, 2])
    h[..., 0, 1] = h[..., 1, 0] = (
        f(x - step, y - step)
        + f(x + step, y + step)
        - f(x + step, y - step)
        - f(x - step, y + step)
    ) / (4 * step**2)
    h[..., 0, 0] = (f(x + step, y) + f(x - step, y) - 2 * f(x, y)) / (step**2)
    h[..., 1, 1] = (f(x, y + step) + f(x, y - step) - 2 * f(x, y)) / (step**2)
    return h

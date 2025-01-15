import numpy as np

from janhke_emde.functions import diff, hessian


def find_critical_points(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, f, threshold: float = 1e-6
) -> np.ndarray:
    """
    Find critical points on a grid by locating points where both partial derivatives are close to zero.
    """
    dx = diff(f, x, y, 1, 0)
    dy = diff(f, x, y, 0, 1)

    dx_zeros = np.abs(dx) < threshold
    dy_zeros = np.abs(dy) < threshold

    saddle_indices = dx_zeros & dy_zeros

    potential_saddles = np.column_stack((
        x[saddle_indices],
        y[saddle_indices],
        z[saddle_indices],
    ))

    return potential_saddles


def principal_curvatures(
    x: float, y: float, f
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Calculate principal curvatures and directions at point (x,y).
    """
    H = hessian(f, x, y)

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    idx = np.argsort(np.abs(eigenvalues))[::-1]
    k1, k2 = eigenvalues[idx]
    v1, v2 = eigenvectors[:, idx].T

    return k1, k2, v1, v2

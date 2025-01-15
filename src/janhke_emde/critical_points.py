import numpy as np

from janhke_emde.config import VisualizationConfig
from janhke_emde.functions import diff, hessian, unique_tolerance


def find_critical_points(
    config: VisualizationConfig,
    iterations=20,
    threshold: float = 1e-6,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find critical points on a grid by locating points where both partial derivatives are close to zero.
    """

    func = config.func

    x, y = np.meshgrid(
        np.linspace(config.bounds.xl, config.bounds.xu, config.critical_points),
        np.linspace(config.bounds.yl, config.bounds.yu, config.critical_points),
    )

    if config.randomize_critical_grid:
        x += np.random.randn(*x.shape)
        y += np.random.randn(*y.shape)

    for it in range(iterations):
        H_1 = np.linalg.pinv(hessian(func, x, y), hermitian=True)

        dx = diff(func, x, y, 1, 0)  # N x N
        dy = diff(func, x, y, 0, 1)  # N x N
        grad = np.stack((dx, dy), axis=-1)  # N x N x 2

        delta = np.matvec(H_1, grad)
        x -= delta[:, :, 0]
        y -= delta[:, :, 1]
    z = func(x, y)
    pts = np.stack((x, y, z), axis=-1)
    pts = pts[
        ((config.bounds.xl <= x) & (x <= config.bounds.xu))
        & ((config.bounds.yl <= y) & (y <= config.bounds.yu))
        & ((config.bounds.zl <= z) & (z <= config.bounds.zu))
    ].reshape(-1, 3)
    pts = unique_tolerance(pts, tol)

    x, y = pts[:, 0], pts[:, 1]
    dx = diff(func, x, y, 1, 0)
    dy = diff(func, x, y, 0, 1)
    pts = pts[dx**2 + dy**2 < threshold**2]

    return pts[:, 0], pts[:, 1], pts[:, 2]


def principal_curvatures(f, x, y):
    """
    Calculate principal curvatures and directions at point (x,y).
    """
    H = hessian(f, x, y)

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    k1, k2 = eigenvalues.T
    v1, v2 = eigenvectors[:, 0, :], eigenvectors[:, 1, :]

    return k1, k2, v1, v2

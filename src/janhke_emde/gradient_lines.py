import numpy as np

from janhke_emde.config import VisualizationConfig
from janhke_emde.functions import diff


def gradient_lines(
    config: VisualizationConfig, starting_pts: np.ndarray, down: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    a = starting_pts

    def do_step(p: np.ndarray, e) -> tuple[np.ndarray, float]:
        grad = np.column_stack((
            diff(config.func, p[:, 0], p[:, 1], 1, 0),
            diff(config.func, p[:, 0], p[:, 1], 0, 1),
        ))
        grad_norm_2 = np.sum(grad**2, axis=1)

        e = config.gradient_gamma * e + (1 - config.gradient_gamma) * grad_norm_2
        step = config.gradient_alpha * grad / np.sqrt(e + 1e-8)[:, None]
        if down:
            step *= -1
        return (p + step, e)

    pts = [a]
    b, e = do_step(a, np.zeros_like(a[:, 0]))
    for _ in range(config.gradient_maxiter - 1):
        pts.append(b)
        a, (b, e) = b, do_step(b, e)
    pts.append(b)

    pts = np.stack(pts, axis=1)
    z = config.func(pts[..., 0], pts[..., 1])
    mask = config.bounds.in_bounds(pts[..., 0], pts[..., 1], z)

    last_valid = np.argmax(mask[:, ::-1], axis=1)
    last_valid = mask.shape[1] - 1 - last_valid

    row_indices = np.arange(pts.shape[0])[:, None]
    col_indices = np.arange(pts.shape[1])[None, :]

    replace_mask = col_indices > last_valid[:, np.newaxis]
    last_valid_pts = pts[row_indices, last_valid[:, np.newaxis]]
    pts = np.where(replace_mask[..., np.newaxis], last_valid_pts, pts)

    return pts, last_valid

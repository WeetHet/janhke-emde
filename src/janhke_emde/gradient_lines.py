import numpy as np
from janhke_emde.functions import diff
from janhke_emde.config import VisualizationConfig


def gradient_line(
    config: VisualizationConfig,
    sx: float,
    sy: float,
) -> np.ndarray:
    a = np.array((sx, sy))
    pts = [a]

    def do_step(p: np.ndarray, e: float) -> tuple[np.ndarray, float]:
        grad = np.array(
            (diff(config.func, p[0], p[1], 1, 0), diff(config.func, p[0], p[1], 0, 1))
        )
        e = float(
            config.gradient_gamma * e + (1 - config.gradient_gamma) * np.sum(grad**2)
        )
        return (p + config.gradient_alpha * grad / np.sqrt(e + 1e-8), e)

    (b, e) = do_step(a, 0)
    it = 1
    while (
        config.bounds.in_bounds(b[0], b[1], config.func(b[0], b[1]))
        and it != config.gradient_maxiter
    ):
        pts.append(b)
        a, (b, e) = b, do_step(b, e)
        it += 1

    pts.append(b)
    return np.array(pts)

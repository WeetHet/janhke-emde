import numpy as np  # noqa: F401
from scipy.special import gamma, zeta

from janhke_emde.config import Bounds3D, VisualizationConfig, VisualizationConfigBuilder
from janhke_emde.critical_points import find_critical_points, principal_curvatures
from janhke_emde.visualization import visualize_surface


def gamma2d(x, y):
    """Compute the absolute value of the gamma function for complex input x + yi."""
    return np.abs(gamma(x + y * 1j))


def zeta2d(x, y):
    """Compute the absolute value of the zeta function for complex input x + yi."""
    return np.abs(zeta(x + y * 1j))


__all__ = [
    "VisualizationConfig",
    "VisualizationConfigBuilder",
    "Bounds3D",
    "visualize_surface",
    "find_critical_points",
    "principal_curvatures",
]


def build_generic_config(bounds: Bounds3D, func):
    return (
        VisualizationConfigBuilder()
        .with_bounds(bounds)
        .with_function(func)
        .with_mesh_points(1000)
        .with_critical_points(200)
        .with_level_params(0.2, 5.0, 0.2)
        .with_gradient_params(gamma=0.9, alpha=0.001, maxiter=10000)
        .with_gradient_points(20)
        .with_log_steps()
        .build()
    )


def run_with_zeta2d():
    bounds = Bounds3D(xl=-4, xu=4, yl=-30, yu=30, zl=0, zu=5)
    config = build_generic_config(bounds, zeta2d)

    visualize_surface(config)


def run_with_gamma2d():
    bounds = Bounds3D(xl=-6, xu=4, yl=-3, yu=3, zl=0, zu=5)
    config = build_generic_config(bounds, gamma2d)

    visualize_surface(config)


def main():
    run_with_gamma2d()
    # run_with_zeta2d()

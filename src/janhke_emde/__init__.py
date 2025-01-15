import numpy as np  # noqa: F401

from janhke_emde.config import Bounds3D, VisualizationConfig, VisualizationConfigBuilder
from janhke_emde.critical_points import find_critical_points, principal_curvatures
from janhke_emde.functions import gamma2d
from janhke_emde.visualization import visualize_surface

__all__ = [
    "VisualizationConfig",
    "VisualizationConfigBuilder",
    "Bounds3D",
    "visualize_surface",
    "find_critical_points",
    "principal_curvatures",
]


def main():
    bounds = Bounds3D(xl=-6, xu=4, yl=-3, yu=3, zl=0, zu=5)

    config = (
        VisualizationConfigBuilder()
        .with_bounds(bounds)
        .with_function(gamma2d)
        .with_mesh_points(1000)
        .with_critical_points(200)
        .with_level_params(0.2, 5.0, 0.2)
        .with_gradient_params(gamma=0.9, alpha=0.001, maxiter=10000)
        .with_gradient_points(20)
        .with_log_steps()
        .build()
    )

    visualize_surface(config)

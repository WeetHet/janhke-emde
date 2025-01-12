from janhke_emde.config import VisualizationConfig, VisualizationConfigBuilder, Bounds3D
from janhke_emde.functions import gamma2d
from janhke_emde.visualization import visualize_surface

__all__ = [
    "VisualizationConfig",
    "VisualizationConfigBuilder",
    "Bounds3D",
    "visualize_surface"
]

def main():
    bounds = Bounds3D(
        xl=-6, xu=4,
        yl=-3, yu=3,
        zl=0, zu=5
    )

    config = (VisualizationConfigBuilder()
        .with_bounds(bounds)
        .with_function(gamma2d)
        .with_mesh_points(1000)
        .with_level_params(0.2, 5.0, 0.2)
        .with_gradient_params(gamma=0.9, alpha=0.01, maxiter=100000)
        .with_gradient_points(20)
        .with_log_steps(True)
        .build())

    visualize_surface(config)

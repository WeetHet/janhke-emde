from dataclasses import dataclass
from typing import Callable

@dataclass
class Bounds3D:
    xl: float
    xu: float
    yl: float
    yu: float
    zl: float
    zu: float

    def in_bounds(self, x: float, y: float, z: float) -> bool:
        return (self.xl <= x <= self.xu and
                self.yl <= y <= self.yu and
                self.zl <= z <= self.zu)

@dataclass
class VisualizationConfig:
    bounds: Bounds3D
    mesh_points: int
    level_start: float
    level_end: float
    level_step: float
    gradient_gamma: float
    gradient_alpha: float
    gradient_maxiter: float
    gradient_points: int
    func: Callable
    log_steps: bool

class VisualizationConfigBuilder:
    def with_bounds(self, bounds: Bounds3D) -> 'VisualizationConfigBuilder':
        self.bounds = bounds
        return self

    def with_mesh_points(self, points: int) -> 'VisualizationConfigBuilder':
        self.mesh_points = points
        return self

    def with_level_params(self, start: float, end: float, step: float) -> 'VisualizationConfigBuilder':
        self.level_start = start
        self.level_end = end
        self.level_step = step
        return self

    def with_gradient_params(self, gamma: float, alpha: float, maxiter: int) -> 'VisualizationConfigBuilder':
        self.gradient_gamma = gamma
        self.gradient_alpha = alpha
        self.gradient_maxiter = maxiter
        return self

    def with_gradient_points(self, points: int) -> 'VisualizationConfigBuilder':
        self.gradient_points = points
        return self

    def with_function(self, func: Callable[[float, float], float]) -> 'VisualizationConfigBuilder':
        self.func = func
        return self

    def with_log_steps(self, log_steps: bool) -> 'VisualizationConfigBuilder':
        self.log_steps = log_steps
        return self

    def build(self) -> VisualizationConfig:
        required_attrs = ['bounds', 'func', 'mesh_points',
                         'level_start', 'level_end', 'level_step',
                         'gradient_gamma', 'gradient_alpha', 'gradient_maxiter',
                         'gradient_points']

        for attr in required_attrs:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                raise ValueError(f"{attr} must be set and be not None")

        return VisualizationConfig(
            bounds=self.bounds,
            func=self.func,
            mesh_points=self.mesh_points,
            level_start=self.level_start,
            level_end=self.level_end,
            level_step=self.level_step,
            gradient_gamma=self.gradient_gamma,
            gradient_alpha=self.gradient_alpha,
            gradient_maxiter=self.gradient_maxiter,
            gradient_points=self.gradient_points,
            log_steps=getattr(self, "log_steps", False)
        )

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

    def in_bounds(self, x, y, z):
        return (
            ((self.xl <= x) & (x <= self.xu))
            & ((self.yl <= y) & (y <= self.yu))
            & ((self.zl <= z) & (z <= self.zu))
        )


@dataclass
class VisualizationConfig:
    bounds: Bounds3D
    mesh_points: int
    critical_points: int
    randomize_critical_grid: bool
    level_start: float
    level_end: float
    level_step: float
    gradient_gamma: float
    gradient_alpha: float
    gradient_iter: int
    gradient_points: int
    func: Callable
    log_steps: bool
    debug: bool
    bounding_box: bool = False
    z_cap_move: float = 0.0


class VisualizationConfigBuilder:
    randomize_critical_grid: bool = False

    def with_bounds(self, bounds: Bounds3D) -> "VisualizationConfigBuilder":
        self.bounds = bounds
        return self

    def with_mesh_points(self, points: int) -> "VisualizationConfigBuilder":
        self.mesh_points = points
        return self

    def with_critical_points(self, points: int) -> "VisualizationConfigBuilder":
        self.critical_points = points
        return self

    def with_randomize_critical_grid(
        self, randomize_critical_grid: bool
    ) -> "VisualizationConfigBuilder":
        self.randomize_critical_grid = randomize_critical_grid
        return self

    def with_level_params(
        self, start: float, end: float, step: float
    ) -> "VisualizationConfigBuilder":
        self.level_start = start
        self.level_end = end
        self.level_step = step
        return self

    def with_gradient_params(
        self, gamma: float, alpha: float, maxiter: int
    ) -> "VisualizationConfigBuilder":
        self.gradient_gamma = gamma
        self.gradient_alpha = alpha
        self.gradient_iter = maxiter
        return self

    def with_gradient_points(self, points: int) -> "VisualizationConfigBuilder":
        self.gradient_points = points
        return self

    def with_function(
        self, func: Callable[[float, float], float]
    ) -> "VisualizationConfigBuilder":
        self.func = func
        return self

    def with_log_steps(self) -> "VisualizationConfigBuilder":
        self.log_steps = True
        return self

    def with_debug(
        self,
    ) -> "VisualizationConfigBuilder":
        self.debug = True
        return self

    def with_bounding_box(self, enable: bool = True) -> "VisualizationConfigBuilder":
        self.bounding_box = enable
        return self

    def with_z_cap_move(self, move_by: float) -> "VisualizationConfigBuilder":
        self.z_cap_move = move_by
        return self

    def build(self) -> VisualizationConfig:
        required_attrs = [
            "bounds",
            "func",
            "mesh_points",
            "critical_points",
            "randomize_critical_grid",
            "level_start",
            "level_end",
            "level_step",
            "gradient_gamma",
            "gradient_alpha",
            "gradient_iter",
            "gradient_points",
        ]

        for attr in required_attrs:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                raise ValueError(f"{attr} must be set and be not None")

        return VisualizationConfig(
            bounds=self.bounds,
            func=self.func,
            mesh_points=self.mesh_points,
            critical_points=self.critical_points,
            randomize_critical_grid=self.randomize_critical_grid,
            level_start=self.level_start,
            level_end=self.level_end,
            level_step=self.level_step,
            gradient_gamma=self.gradient_gamma,
            gradient_alpha=self.gradient_alpha,
            gradient_iter=self.gradient_iter,
            gradient_points=self.gradient_points,
            log_steps=getattr(self, "log_steps", False),
            debug=getattr(self, "debug", False),
            bounding_box=getattr(self, "bounding_box", False),
            z_cap_move=getattr(self, "z_cap_move", 0.0),
        )

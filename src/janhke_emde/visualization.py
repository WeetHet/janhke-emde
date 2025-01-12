import numpy as np
import pyvista as pv
from janhke_emde.config import VisualizationConfig
from janhke_emde.functions import gamma2d, diff
from janhke_emde.level_curves import find_level_segments, decompose_levels_as_cycles_and_paths
from janhke_emde.gradient_lines import gradient_line

def print_with_config(config: VisualizationConfig, *args, **kwargs):
    if config.log_steps:
        print(*args, **kwargs)

def create_mesh_grid(config: VisualizationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print_with_config(config, "Creating meshgrid...")
    x, y = np.meshgrid(
        np.linspace(config.bounds.xl, config.bounds.xu, config.mesh_points),
        np.linspace(config.bounds.yl, config.bounds.yu, config.mesh_points)
    )
    print_with_config(config, "Computing function values and clipping...")
    z = np.clip(config.func(x, y), config.bounds.zl, config.bounds.zu)
    return x, y, z

def setup_surface(plotter: pv.Plotter, x: np.ndarray, y: np.ndarray, z: np.ndarray, config: VisualizationConfig) -> None:
    print_with_config(config, "Setting up 3D plot...")
    grid = pv.StructuredGrid(x, y, z)
    print_with_config(config, "Plotting surface...")
    plotter.add_mesh(grid, color="white", show_edges=False, lighting=False)

def plot_level_curves(plotter: pv.Plotter, x: np.ndarray, y: np.ndarray, z: np.ndarray, buffer: np.ndarray, config: VisualizationConfig) -> None:
    print_with_config(config, "Working on level ", end="")
    for level in np.arange(config.level_start, config.level_end, config.level_step):
        print_with_config(config, f"{level:.1f}", end="; ", flush=True)
        find_level_segments(x, y, z, level, buffer)

        nan_2d_mask = np.isnan(buffer)
        non_nan_mask = ~np.any(nan_2d_mask, axis=(1, 2))

        segments = buffer[non_nan_mask]
        cycles, paths, _ = decompose_levels_as_cycles_and_paths(segments)
        for cycle in cycles + paths:
            points = cycle[:, :2]
            dx = diff(gamma2d, points[:, 0], points[:, 1], 1, 0)
            dy = diff(gamma2d, points[:, 0], points[:, 1], 0, 1)
            gradients = np.column_stack((dx, dy, np.zeros_like(dx)))

            gradient_magnitudes = np.linalg.norm(gradients, axis=1)
            base_offset = 0.01
            adaptive_scales = 1 / (1 + gradient_magnitudes)
            offsets = (
                -base_offset
                * adaptive_scales[:, np.newaxis]
                * gradients
                / (gradient_magnitudes[:, np.newaxis] + 1e-8)
            )

            moved_points = cycle + offsets

            curve = pv.lines_from_points(moved_points)
            plotter.add_mesh(curve, color="black", line_width=2)
    print_with_config(config)

def plot_gradient_lines(plotter: pv.Plotter, config: VisualizationConfig) -> None:
    print_with_config(config, "Drawing gradient lines...")

    points_per_side = config.gradient_points
    left_points = np.column_stack((
        np.full(points_per_side, config.bounds.xl),
        np.linspace(config.bounds.yl, config.bounds.yu, points_per_side)
    ))
    right_points = np.column_stack((
        np.full(points_per_side, config.bounds.xu),
        np.linspace(config.bounds.yl, config.bounds.yu, points_per_side)
    ))
    bottom_points = np.column_stack((
        np.linspace(config.bounds.xl, config.bounds.xu, points_per_side),
        np.full(points_per_side, config.bounds.yl)
    ))
    top_points = np.column_stack((
        np.linspace(config.bounds.xl, config.bounds.xu, points_per_side),
        np.full(points_per_side, config.bounds.yu)
    ))

    starting_points = np.vstack((left_points, right_points, bottom_points, top_points))

    for start_pt in starting_points:
        gradient_pts = gradient_line(
            config,
            start_pt[0],
            start_pt[1],
        )
        gradient_z = gamma2d(gradient_pts[:, 0], gradient_pts[:, 1])
        points_3d = np.column_stack((gradient_pts, gradient_z))

        dx = diff(gamma2d, gradient_pts[:, 0], gradient_pts[:, 1], 1, 0)
        dy = diff(gamma2d, gradient_pts[:, 0], gradient_pts[:, 1], 0, 1)
        gradients = np.column_stack((dx, dy, np.zeros_like(dx)))

        gradient_magnitudes = np.linalg.norm(gradients, axis=1)
        base_offset = 0.01
        adaptive_scales = 1 / (1 + gradient_magnitudes)
        offsets = (
            -base_offset
            * adaptive_scales[:, np.newaxis]
            * gradients
            / (gradient_magnitudes[:, np.newaxis] + 1e-8)
        )

        moved_points = np.clip(
            points_3d + offsets,
            np.array([config.bounds.xl, config.bounds.yl, 0]),
            np.array([config.bounds.xu, config.bounds.yu, config.level_end]),
        )

        zl_eps = config.level_end + 2e-3
        if len(moved_points) > 0 and abs(moved_points[-1, 2] - config.level_end) < 0.5:
            moved_points[-1, 2] = zl_eps
            n_points = min(5, len(moved_points))
            for i in range(n_points):
                idx = -1 - i
                t = i / n_points
                moved_points[idx, 2] = zl_eps * (1 - t) + moved_points[idx, 2] * t

        gradient_curve = pv.lines_from_points(moved_points)
        plotter.add_mesh(gradient_curve, color="black", line_width=2)

def plot_upper_caps(plotter: pv.Plotter, x: np.ndarray, y: np.ndarray, z: np.ndarray, buffer: np.ndarray, config: VisualizationConfig) -> None:
    print_with_config(config, "Drawing upper caps...")
    find_level_segments(x, y, z, config.level_end - 1e-3, buffer)
    nan_2d_mask = np.isnan(buffer)
    non_nan_mask = ~np.any(nan_2d_mask, axis=(1, 2))

    segments = buffer[non_nan_mask]
    cycles, paths, _ = decompose_levels_as_cycles_and_paths(segments)

    print_with_config(config, "Filling in the cycles...")
    for cycle in filter(lambda c: len(c) > 1, cycles):
        lifted_cycle = cycle.copy()
        lifted_cycle[:, 2] += 5e-3
        poly = pv.PolyData(lifted_cycle)
        poly = poly.delaunay_2d()
        plotter.add_mesh(poly, color="black")

    print_with_config(config, "Filling in the paths...")
    for path in filter(lambda p: len(p) > 1, paths):
        lifted_path = path.copy()
        lifted_path[:, 2] += 5e-3
        poly = pv.PolyData(lifted_path)
        poly = poly.delaunay_2d()
        plotter.add_mesh(poly, color="black")

def visualize_surface(config: VisualizationConfig) -> None:
    """Main entry point for visualizing the Jahnke-Emde surface with level curves and gradient lines."""
    plotter = pv.Plotter()
    x, y, z = create_mesh_grid(config)
    setup_surface(plotter, x, y, z, config)

    buffer = np.zeros((4 * x.shape[0] * y.shape[1], 2, 3))
    plot_level_curves(plotter, x, y, z, buffer, config)
    plot_upper_caps(plotter, x, y, z, buffer, config)
    plot_gradient_lines(plotter, config)

    print_with_config(config, "Showing plot...")
    plotter.view_isometric() # type: ignore
    plotter.show()

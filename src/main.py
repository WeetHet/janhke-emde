from scipy.special import gamma
from typing import Callable, Optional
import numpy as np
import numpy.typing as nt
from functools import lru_cache
from typing import no_type_check
import pyvista as pv
import networkx as nx
from numba import jit, prange

def gamma2d(x, y):
    return np.abs(gamma(x + y * 1j))

@jit(nopython=True)
def _get_level_point(a: np.ndarray, b: np.ndarray, level: float) -> Optional[np.ndarray]:
    if a[2] > b[2]:
        a, b = b, a
    if a[2] <= level <= b[2]:
        vec = b - a
        return a + vec / np.linalg.norm(vec) * (level - a[2])
    return np.full(3, np.nan)

@jit(nopython=True)
def all_levels(
    pts: np.ndarray,
    level: float,
):
    all_level_points = np.zeros((len(pts), 3))
    for i in prange(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        level_point = _get_level_point(a, b, level)
        all_level_points[i] = level_point
    return all_level_points

@jit(nopython=True, parallel=True)
def find_level_segments(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    level: float,
    buffer: np.ndarray
) -> np.ndarray:
    x_iter, y_iter = x.shape[0]-1, y.shape[1]-1

    for i in prange(x_iter):
        for j in prange(y_iter):
            pts = np.empty((4, 3))
            for n, (di, dj) in enumerate(((0, 0), (0, 1), (1, 1), (1, 0))):
                pts[n] = np.array((
                    x[i+di,j+dj],
                    y[i+di,j+dj],
                    z[i+di,j+dj]
                ))
            all_level_points = all_levels(pts, level)
            xs, ys, zs = all_level_points[:, 0], all_level_points[:, 1], all_level_points[:, 2]
            mask = ~(np.isnan(xs) | np.isnan(ys) | np.isnan(zs))
            level_points = all_level_points[mask]

            count = 4 * (i * y_iter + j)
            for idx in range(len(level_points)):
                buffer[count + idx][0] = level_points[idx]
                buffer[count + idx][1] = level_points[(idx + 1) % len(level_points)]
            for idx in range(len(level_points), 4):
                buffer[count + idx] = np.full((2, 3), np.nan)

    return buffer

def diff(f, x, y, dx, dy, step=1e-6):
    return (f(x + dx * step, y + dy * step) - f(x, y)) / step

def gradient_line(
    f: Callable[[float, float], float],
    in_bounding_box: Callable[[Callable[[float, float], float], float, float], bool],
    sx: float, sy: float,
    step = 1e-1,
    maxiter: int = -1
) -> np.ndarray:
    a = np.array((sx, sy))
    pts = [a]

    grad = np.array((diff(f, sx, sy, 1, 0), diff(f, sx, sy, 0, 1)))
    grad_norm = grad / np.linalg.norm(grad)

    b = a + grad_norm * step
    it = 1
    while in_bounding_box(f, *b) and it != maxiter:
        pts.append(b)

        new_grad = np.array((diff(f, *b, 1, 0), diff(f, *b, 0, 1)))
        new_grad_norm = new_grad / np.linalg.norm(new_grad)
        a, b = b, b + new_grad_norm * step
        it += 1

    return np.array(pts)

def decompose_levels_as_cycles_and_paths(
    segments: np.ndarray,
    tolerance: float = 1e-6
) -> tuple[list[np.ndarray], list[np.ndarray], nx.Graph]:
    points = segments.reshape(-1, 3)
    unique_points = np.unique(points, axis=0)

    G = nx.Graph()

    for i, point in enumerate(unique_points):
        G.add_node(i, pos=point)

    for segment in segments:
        i1 = np.abs(unique_points - segment[0]).sum(axis=1).argmin()
        i2 = np.abs(unique_points - segment[1]).sum(axis=1).argmin()
        G.add_edge(i1, i2)

    cycles = []
    for cycle in nx.cycle_basis(G):
        cycle_points = np.array([unique_points[i] for i in cycle])
        cycles.append(cycle_points)

    paths = []
    for component in nx.connected_components(G):
        if len(component) > 1:
            subgraph = G.subgraph(component)
            if not any(all(v in cycle for cycle in cycles) for v in subgraph.nodes()):
                degree_one_vertices = [v for v in subgraph.nodes() if subgraph.degree(v) == 1]
                if len(degree_one_vertices) >= 2:
                    path = nx.shortest_path(subgraph, degree_one_vertices[0], degree_one_vertices[1])
                    path_points = np.array([unique_points[i] for i in path])
                    paths.append(path_points)

    return cycles, paths, G

def plot_surface_with_levels():
    print("Creating meshgrid...")
    x, y = np.meshgrid(np.linspace(-6, 4, 1000), np.linspace(-3, 3, 1000))

    print("Computing gamma function values and clipping...")
    z = np.clip(gamma2d(x, y), 0, 5)

    print("Setting up 3D plot...")
    grid = pv.StructuredGrid(x, y, z)

    print("Creating plotter...")
    plotter = pv.Plotter()

    print("Plotting surface...")
    plotter.add_mesh(grid, color='white', show_edges=False, lighting=False)

    buffer =np.zeros((4 * x.shape[0] * y.shape[1], 2, 3))

    print("Working on level ", end="")
    for level in np.arange(0.2, 5.2, 0.2):
        print(f"{level:.1f}", end=", " if level != 5.0 else "\n", flush=True)
        find_level_segments(x, y, z, level, buffer)

        nan_2d_mask = np.isnan(buffer)
        non_nan_mask = ~np.any(nan_2d_mask, axis=(1,2))

        segments = buffer[non_nan_mask]
        cycles, paths, graph = decompose_levels_as_cycles_and_paths(segments)
        for cycle in cycles + paths:
            points = cycle[:, :2]
            dx = diff(gamma2d, points[:,0], points[:,1], 1, 0)
            dy = diff(gamma2d, points[:,0], points[:,1], 0, 1)
            gradients = np.column_stack((dx, dy, np.zeros_like(dx)))

            gradient_magnitudes = np.linalg.norm(gradients, axis=1)
            base_offset = 0.01
            adaptive_scales = 1 / (1 + gradient_magnitudes)
            offsets = -base_offset * adaptive_scales[:, np.newaxis] * gradients / (gradient_magnitudes[:, np.newaxis] + 1e-8)

            moved_points = cycle + offsets

            curve = pv.lines_from_points(moved_points)
            plotter.add_mesh(curve, color='black', line_width=2)

    print("Drawing gradient lines...")
    def in_bounds(f, x, y):
        return -6 <= x <= 4 and -3 <= y <= 3 and 0 <= f(x, y) <= 5

    # Define the boundaries
    x_min, x_max = -6, 4
    y_min, y_max = -3, 3

    # Generate starting points along each edge
    left_points = np.column_stack((np.full(20, x_min), np.linspace(y_min, y_max, 20)))
    right_points = np.column_stack((np.full(20, x_max), np.linspace(y_min, y_max, 20)))
    bottom_points = np.column_stack((np.linspace(x_min, x_max, 20), np.full(20, y_min)))
    top_points = np.column_stack((np.linspace(x_min, x_max, 20), np.full(20, y_max)))

    starting_points = np.vstack((left_points, right_points, bottom_points, top_points))

    # Draw gradient lines from each starting point
    for start_pt in starting_points:
        gradient_pts = gradient_line(gamma2d, in_bounds, start_pt[0], start_pt[1], maxiter=10000)
        gradient_z = gamma2d(gradient_pts[:,0], gradient_pts[:,1])
        points_3d = np.column_stack((gradient_pts, gradient_z))

        dx = diff(gamma2d, gradient_pts[:,0], gradient_pts[:,1], 1, 0)
        dy = diff(gamma2d, gradient_pts[:,0], gradient_pts[:,1], 0, 1)
        gradients = np.column_stack((dx, dy, np.zeros_like(dx)))

        gradient_magnitudes = np.linalg.norm(gradients, axis=1)
        base_offset = 0.01
        adaptive_scales = 1 / (1 + gradient_magnitudes)
        offsets = -base_offset * adaptive_scales[:, np.newaxis] * gradients / (gradient_magnitudes[:, np.newaxis] + 1e-8)

        moved_points = points_3d + offsets

        gradient_curve = pv.lines_from_points(moved_points)
        plotter.add_mesh(gradient_curve, color='black', line_width=2)

    print("Showing plot...")
    plotter.view_isometric() # type: ignore
    plotter.show()

if __name__ == "__main__":
    plot_surface_with_levels()

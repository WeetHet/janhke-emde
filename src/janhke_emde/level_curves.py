import numpy as np
import networkx as nx
from numba import jit, prange


@jit(nopython=True)
def _get_level_point(a: np.ndarray, b: np.ndarray, level: float) -> np.ndarray:
    if a[2] > b[2]:
        a, b = b, a
    if a[2] <= level <= b[2]:
        vec = b - a
        return a + vec / np.linalg.norm(vec) * (level - a[2])
    return np.full(3, np.nan)


@jit(nopython=True)
def all_levels(pts: np.ndarray, level: float):
    all_level_points = np.zeros((len(pts), 3))
    for i in prange(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        level_point = _get_level_point(a, b, level)
        all_level_points[i] = level_point
    return all_level_points


@jit(nopython=True, parallel=True)
def find_level_segments(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, level: float, buffer: np.ndarray
) -> np.ndarray:
    x_iter, y_iter = x.shape[0] - 1, y.shape[1] - 1

    for i in prange(x_iter):
        for j in prange(y_iter):
            pts = np.empty((4, 3))
            for n, (di, dj) in enumerate(((0, 0), (0, 1), (1, 1), (1, 0))):
                pts[n] = np.array(
                    (x[i + di, j + dj], y[i + di, j + dj], z[i + di, j + dj])
                )
            all_level_points = all_levels(pts, level)
            xs, ys, zs = (
                all_level_points[:, 0],
                all_level_points[:, 1],
                all_level_points[:, 2],
            )
            mask = ~(np.isnan(xs) | np.isnan(ys) | np.isnan(zs))
            level_points = all_level_points[mask]

            count = 4 * (i * y_iter + j)
            for idx in range(len(level_points)):
                buffer[count + idx][0] = level_points[idx]
                buffer[count + idx][1] = level_points[(idx + 1) % len(level_points)]
            for idx in range(len(level_points), 4):
                buffer[count + idx] = np.full((2, 3), np.nan)

    return buffer


def decompose_levels_as_cycles_and_paths(
    segments: np.ndarray, tolerance: float = 1e-6
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
                degree_one_vertices = [
                    v for v in subgraph.nodes() if subgraph.degree(v) == 1
                ]
                if len(degree_one_vertices) >= 2:
                    path = nx.shortest_path(
                        subgraph, degree_one_vertices[0], degree_one_vertices[1]
                    )
                    path_points = np.array([unique_points[i] for i in path])
                    paths.append(path_points)

    return cycles, paths, G

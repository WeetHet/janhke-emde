# Janhke-Emde Style Function Visualization

## Features

- Janhke-Emde style function surface style
- Level curves and gradient flow visualization
- Critical point detection
- Customizable visualization parameters

## Usage

```python
from scipy.special import gamma
import numpy as np

from janhke_emde import (
    Bounds3D,
    VisualizationConfigBuilder,
    visualize_surface
)

def gamma2d(x, y):
    """Compute the absolute value of the gamma function for complex input x + yi."""
    return np.abs(gamma(x + y * 1j))

bounds = Bounds3D(xl=-6, xu=4, yl=-3, yu=3, zl=0, zu=5)

config = (
    VisualizationConfigBuilder()
    .with_bounds(bounds)
    .with_function(f)
    .with_mesh_points(1000)
    .with_critical_points(200)
    .with_level_params(0.2, 5.0, 0.2)
    .with_gradient_params(gamma=0.9, alpha=0.001, maxiter=10000)
    .with_gradient_points(20)
    .build()
)

visualize_surface(config)
```

## Configuration Options

- `bounds`: x, y and z bounds for visualization
- `mesh_points`: Number of points per side to use in the surface mesh grid
- `critical_points`: Number of points per side to use when searching for critical points
- `randomize_critical_grid`: Boolean flag to randomize the critical point search grid
- `level_start`: Starting value for level curves
- `level_end`: Ending value for level curves
- `level_step`: Step size between level curves
- `gradient_gamma`: Gradient descent momentum parameter (between 0 and 1)
- `gradient_alpha`: Gradient descent learning rate
- `gradient_iter`: Number of gradient descent iterations
- `gradient_points`: Number of starting points for gradient lines on each boundary
- `func`: Callable function to visualize, takes x,y coordinates and returns z value
- `log_steps`: Boolean flag to enable progress logging

## Requirements

- NumPy
- SciPy
- PyVista
- NetworkX
- Numba

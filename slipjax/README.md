# SlipJAX

A JAX-based implementation of vehicle dynamics models from "Dynamics And Control Of Drifting In Automobiles" (Hindiyeh, 2013), ported from MATLAB.

![Vehicle Steering Trajectory](examples/steering_trajectory_plot.png)

## Overview

SlipJAX is a Python package that translates MATLAB vehicle dynamics models to JAX-compatible Python code. The implementation leverages JAX for GPU acceleration, automatic differentiation, and JIT compilation benefits while maintaining the mathematical accuracy of the original models.

## Features

- **JAX Compatibility**: All computations are JAX-compatible, allowing for GPU acceleration and automatic differentiation
- **JIT Compilation**: Model functions are decorated with `@jit` for faster execution
- **Pure Functional Design**: Models implemented using pure functions for better composability
- **Type Annotations**: Comprehensive type hints using JAX-specific types
- **Batch Processing**: Models support batch processing of multiple vehicle states simultaneously

## Models Implemented

- **Front Tire Dynamics**: Calculates lateral force for the front tire based on slip angle
- **Rear Tire Dynamics**: Calculates longitudinal and lateral forces for the rear tire based on multiple parameters

## Installation

```bash
pip install slipjax
```

For development:

```bash
git clone https://github.com/yourusername/slipjax.git
cd slipjax
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
from slipjax.front_tire_dynamics import calculate_front_tire_lateral_force
from slipjax.rear_tire_dynamics import calculate_rear_tire_forces

# Front tire example
slip_angle = jnp.array(0.1)  # radians
friction_coefficient = jnp.array(0.7)
front_load = jnp.array(3000.0)  # Newtons
cornering_stiffness = jnp.array(50000.0)  # N/rad

lateral_force = calculate_front_tire_lateral_force(
    slip_angle, 
    friction_coefficient, 
    front_load, 
    cornering_stiffness
)

# Rear tire example
longitudinal_velocity = jnp.array(10.0)  # m/s
wheel_velocity = jnp.array(11.0)  # m/s
slip_angle = jnp.array(0.05)  # radians
friction_coefficient = jnp.array(0.7)
sliding_friction_coefficient = jnp.array(0.5)
rear_load = jnp.array(3500.0)  # Newtons
longitudinal_stiffness = jnp.array(40000.0)  # N/unit slip
cornering_stiffness = jnp.array(45000.0)  # N/rad

longitudinal_force, lateral_force = calculate_rear_tire_forces(
    longitudinal_velocity,
    wheel_velocity,
    slip_angle,
    friction_coefficient,
    sliding_friction_coefficient,
    rear_load,
    longitudinal_stiffness,
    cornering_stiffness
)
```

## Model Details

The vehicle dynamics models are based on the mathematical formulations presented in "Dynamics And Control Of Drifting In Automobiles" (Hindiyeh, 2013). The models calculate forces based on:

- **Slip Angle**: Angle between tire's heading and actual direction of travel
- **Longitudinal Slip**: Ratio of wheel velocity to vehicle velocity
- **Friction Models**: Including linear and saturation regions
- **Tire Stiffness Properties**: Cornering and longitudinal stiffness parameters

For detailed model documentation, see the [dynamics documentation](docs/dynamics.md).

## Development

### Testing

```bash
pytest
```

### Code Formatting

```bash
black slipjax tests
isort slipjax tests
```

### Type Checking

```bash
mypy slipjax
```

## License

MIT License

## Citation

If you use SlipJAX in your research, please cite:

```
@software{slipjax2023,
  author = {Your Name},
  title = {SlipJAX: JAX-based Vehicle Dynamics Models},
  year = {2023},
  url = {https://github.com/yourusername/slipjax}
}
```

Original MATLAB model:
```
@phdthesis{hindiyeh2013dynamics,
  title={Dynamics and control of drifting in automobiles},
  author={Hindiyeh, Rami Y},
  year={2013},
  school={Stanford University}
}
```
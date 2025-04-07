"""Parameters and constants for vehicle dynamics models."""

import jax.numpy as jnp

# Tire model parameters
FRONT_CORNERING_STIFFNESS = 900.0  # N/rad
REAR_CORNERING_STIFFNESS = 1000.0  # N/rad
SLIDING_LIMIT_ANGLE = 0.05  # rad

# Vehicle parameters
VEHICLE_MASS = 1500.0  # kg
WHEELBASE = 2.7  # m
FRONT_AXLE_MASS_RATIO = 0.55  # dimensionless
REAR_AXLE_MASS_RATIO = 0.45  # dimensionless
GRAVITY = 9.81  # m/s²

# Derived parameters
FRONT_STATIC_LOAD = VEHICLE_MASS * FRONT_AXLE_MASS_RATIO * GRAVITY  # N
REAR_STATIC_LOAD = VEHICLE_MASS * REAR_AXLE_MASS_RATIO * GRAVITY  # N

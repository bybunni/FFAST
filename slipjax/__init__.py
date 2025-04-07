"""SlipJAX: A JAX implementation of vehicle dynamics models.

This package ports MATLAB vehicle dynamics models to JAX in Python,
especially focusing on tire dynamics and vehicle behavior during drifting.
"""

# Import key components for easier access
from slipjax.models.tire.front import FrontTireDynamics, calculate_front_tire_lateral_force
from slipjax.models.tire.rear import RearTireDynamics, calculate_rear_tire_forces
from slipjax.models.vehicle.dynamics import (
    VehicleDynamics,
    vehicleDynamics,
    VehicleParameters,
    calculate_vehicle_dynamics,
    wrap_to_pi
)

__all__ = [
    # Tire models
    'FrontTireDynamics',
    'RearTireDynamics',
    'calculate_front_tire_lateral_force',
    'calculate_rear_tire_forces',
    
    # Vehicle dynamics
    'VehicleDynamics',
    'vehicleDynamics',
    'VehicleParameters',
    'calculate_vehicle_dynamics',
    'wrap_to_pi'
]

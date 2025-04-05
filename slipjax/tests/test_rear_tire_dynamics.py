"""Tests for rear tire dynamics module."""
from typing import Any

import jax.numpy as jnp
from jax import vmap
import pytest

from slipjax.rear_tire_dynamics import calculate_rear_tire_forces


def test_rear_tire_dynamics_sanity() -> None:
    """Verify that calculate_rear_tire_forces produces expected outputs for basic inputs."""
    # Test parameters
    friction_coefficient = 1.0  # Friction coefficient
    sliding_friction_coefficient = 0.8  # Sliding friction coefficient
    rear_load = 1000.0  # Rear tire load (N)
    longitudinal_stiffness = 5000.0  # Longitudinal stiffness (N/unit slip)
    cornering_stiffness = 1000.0  # Cornering stiffness (N/rad)
    
    # Test case 1: Zero slip case (no longitudinal or lateral forces)
    v_x_zero_slip = 10.0  # Vehicle velocity
    wheel_vx_zero_slip = 10.0  # Wheel velocity matches vehicle (no slip)
    alpha_zero_slip = 0.0  # Zero slip angle
    
    fx_zero_slip, fy_zero_slip = calculate_rear_tire_forces(
        v_x_zero_slip,
        wheel_vx_zero_slip,
        alpha_zero_slip,
        friction_coefficient,
        sliding_friction_coefficient,
        rear_load,
        longitudinal_stiffness,
        cornering_stiffness
    )
    
    assert jnp.isclose(fx_zero_slip, 0.0), f"Expected Fx=0 with zero slip, got {fx_zero_slip}"
    assert jnp.isclose(fy_zero_slip, 0.0), f"Expected Fy=0 with zero slip, got {fy_zero_slip}"
    
    # Test case 2: Pure longitudinal slip (acceleration, no lateral slip)
    v_x_long_slip = 10.0
    wheel_vx_long_slip = 11.0  # 10% slip
    alpha_long_slip = 0.0
    
    fx_long_slip, fy_long_slip = calculate_rear_tire_forces(
        v_x_long_slip,
        wheel_vx_long_slip,
        alpha_long_slip,
        friction_coefficient,
        sliding_friction_coefficient,
        rear_load,
        longitudinal_stiffness,
        cornering_stiffness
    )
    
    assert fx_long_slip > 0, "Expected positive Fx during acceleration"
    assert jnp.isclose(fy_long_slip, 0.0), f"Expected Fy=0 with pure longitudinal slip, got {fy_long_slip}"
    
    # Test case 3: Pure lateral slip (cornering, no longitudinal slip)
    v_x_lat_slip = 10.0
    wheel_vx_lat_slip = 10.0  # No longitudinal slip
    alpha_lat_slip = 0.05  # Small slip angle
    
    fx_lat_slip, fy_lat_slip = calculate_rear_tire_forces(
        v_x_lat_slip,
        wheel_vx_lat_slip,
        alpha_lat_slip,
        friction_coefficient,
        sliding_friction_coefficient,
        rear_load,
        longitudinal_stiffness,
        cornering_stiffness
    )
    
    assert jnp.isclose(fx_lat_slip, 0.0, atol=1e-5), f"Expected Fx=0 with pure lateral slip, got {fx_lat_slip}"
    assert fy_lat_slip < 0, "Expected negative Fy during cornering with positive slip angle"
    
    # Test case 4: Combined slip (both longitudinal and lateral)
    v_x_combined = 10.0
    wheel_vx_combined = 11.0  # 10% slip
    alpha_combined = 0.05  # Small slip angle
    
    fx_combined, fy_combined = calculate_rear_tire_forces(
        v_x_combined,
        wheel_vx_combined,
        alpha_combined,
        friction_coefficient,
        sliding_friction_coefficient,
        rear_load,
        longitudinal_stiffness,
        cornering_stiffness
    )
    
    assert fx_combined > 0, "Expected positive Fx during combined slip with acceleration"
    assert fy_combined < 0, "Expected negative Fy during combined slip with positive slip angle"
    
    # Combined slip should result in lower forces than pure slip due to friction ellipse
    # Not testing exact values as they depend on the specific model parameters


def test_rear_tire_dynamics_edge_cases() -> None:
    """Test rear tire dynamics with edge cases like infinite slip and extreme angles."""
    # Test parameters
    friction_coefficient = 1.0
    sliding_friction_coefficient = 0.8
    rear_load = 1000.0
    longitudinal_stiffness = 5000.0
    cornering_stiffness = 1000.0
    
    # Test case 1: Infinite slip (vehicle stationary, wheels spinning)
    v_x_stationary = 0.0
    wheel_vx_spinning = 10.0
    alpha_spinning = 0.0
    
    fx_infinite_slip, fy_infinite_slip = calculate_rear_tire_forces(
        v_x_stationary,
        wheel_vx_spinning,
        alpha_spinning,
        friction_coefficient,
        sliding_friction_coefficient,
        rear_load,
        longitudinal_stiffness,
        cornering_stiffness
    )
    
    expected_fx_infinite = friction_coefficient * rear_load  # Maximum longitudinal force
    assert jnp.isclose(fx_infinite_slip, expected_fx_infinite), \
        f"Expected Fx={expected_fx_infinite} with infinite slip, got {fx_infinite_slip}"
    assert jnp.isclose(fy_infinite_slip, 0.0), \
        f"Expected Fy=0 with infinite slip, got {fy_infinite_slip}"
    
    # Test case 2: Vehicle moving backward (extreme slip angle > π/2)
    v_x_backward = 10.0
    wheel_vx_backward = 10.0
    alpha_backward = 2.0  # >π/2, should be transformed
    
    fx_backward, fy_backward = calculate_rear_tire_forces(
        v_x_backward,
        wheel_vx_backward,
        alpha_backward,
        friction_coefficient,
        sliding_friction_coefficient,
        rear_load,
        longitudinal_stiffness,
        cornering_stiffness
    )
    
    # According to the model, angles > π/2 are transformed to (π-angle)*sign
    transformed_alpha = (jnp.pi - alpha_backward) * jnp.sign(alpha_backward)
    _, fy_reference = calculate_rear_tire_forces(
        v_x_backward,
        wheel_vx_backward,
        transformed_alpha,
        friction_coefficient,
        sliding_friction_coefficient,
        rear_load,
        longitudinal_stiffness,
        cornering_stiffness
    )
    
    assert jnp.isclose(fy_backward, fy_reference), \
        f"Expected transformation of extreme angles to work correctly"


def test_rear_tire_dynamics_batch_processing() -> None:
    """Test the batch processing capability for massively parallel simulation."""
    # Define a batch of slip angles
    batch_size = 100
    slip_angles = jnp.linspace(-0.5, 0.5, batch_size)
    
    # Constant parameters for all batch elements
    longitudinal_velocity = jnp.ones_like(slip_angles) * 10.0
    wheel_velocity = jnp.ones_like(slip_angles) * 10.0  # No longitudinal slip
    friction_coefficient = 1.0
    sliding_friction_coefficient = 0.8
    rear_load = 1000.0
    longitudinal_stiffness = 5000.0
    cornering_stiffness = 1000.0
    
    # Create a vectorized version of the function that operates on the slip angle
    vectorized_tire_model = vmap(
        calculate_rear_tire_forces, 
        in_axes=(0, 0, 0, None, None, None, None, None)
    )
    
    # Process all slip angles in a single operation
    batch_fx, batch_fy = vectorized_tire_model(
        longitudinal_velocity,
        wheel_velocity,
        slip_angles,
        friction_coefficient,
        sliding_friction_coefficient,
        rear_load,
        longitudinal_stiffness,
        cornering_stiffness
    )
    
    # Verify batch shape matches input shape
    assert batch_fx.shape == slip_angles.shape, f"Expected Fx shape {slip_angles.shape}, got {batch_fx.shape}"
    assert batch_fy.shape == slip_angles.shape, f"Expected Fy shape {slip_angles.shape}, got {batch_fy.shape}"
    
    # Verify that lateral forces have the expected antisymmetric behavior
    # (for pure lateral slip with no longitudinal slip, Fy should be antisymmetric)
    midpoint = batch_size // 2
    positive_angles = slip_angles[midpoint:]
    negative_angles = slip_angles[:midpoint]
    positive_fy = batch_fy[midpoint:]
    negative_fy = batch_fy[:midpoint]
    
    # Lateral forces should be antisymmetric: Fy(-α) = -Fy(α)
    for i in range(len(negative_angles)):
        neg_idx = midpoint - i - 1  # Index from the end of negative angles
        pos_idx = i  # Matching positive angle index
        if pos_idx < len(positive_angles):
            assert jnp.isclose(negative_fy[neg_idx], -positive_fy[pos_idx], rtol=1e-4), \
                f"Force symmetry violated at angles {negative_angles[neg_idx]} and {positive_angles[pos_idx]}"

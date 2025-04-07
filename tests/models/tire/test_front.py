"""Tests for front tire dynamics module."""
from typing import Any

import jax.numpy as jnp
from jax import vmap
import pytest

from slipjax.models.tire.front import calculate_front_tire_lateral_force, FrontTireDynamics


def test_front_tire_dynamics_sanity() -> None:
    """Verify that front_tire_dynamics produces expected outputs for basic inputs."""
    # Test parameters
    mu = 1.0  # Friction coefficient
    load_f = 1000.0  # Front tire load (N)
    C_alpha = 1000.0  # Cornering stiffness (N/rad)
    
    # Create model instance for testing class-based interface
    model = FrontTireDynamics(cornering_stiffness=C_alpha)
    
    # Test with zero slip angle
    alpha_zero = 0.0
    # Test both the class-based interface and the legacy function
    # Use the JIT-compiled versions for the class interface
    Fy_zero_cls = model.jit_call(alpha_zero, jnp.zeros_like(alpha_zero), load_f, mu)
    Fy_zero = calculate_front_tire_lateral_force(alpha_zero, mu, load_f, C_alpha)
    
    assert jnp.isclose(Fy_zero, 0.0), f"Expected Fy=0 with zero slip angle, got {Fy_zero}"
    assert jnp.isclose(Fy_zero_cls, 0.0), f"Expected Fy=0 with zero slip angle (class), got {Fy_zero_cls}"
    
    # Test with small slip angle (in linear region)
    alpha_small = 0.05  # ~2.86 degrees
    Fy_small = calculate_front_tire_lateral_force(alpha_small, mu, load_f, C_alpha)
    Fy_small_cls = model.jit_call(alpha_small, jnp.zeros_like(alpha_small), load_f, mu)
    
    # In linear region, Fy ≈ -C_alpha * tan(alpha)
    expected_Fy_small = -C_alpha * jnp.tan(alpha_small)
    assert jnp.isclose(Fy_small, expected_Fy_small, rtol=0.1), \
        f"Expected Fy≈{expected_Fy_small} with small slip angle, got {Fy_small}"
    assert jnp.isclose(Fy_small_cls, expected_Fy_small, rtol=0.1), \
        f"Expected Fy≈{expected_Fy_small} with small slip angle (class), got {Fy_small_cls}"
    
    # Test with large slip angle (in saturation region)
    alpha_sl = jnp.arctan(3 * mu * load_f / C_alpha)
    alpha_large = 1.5  # ~86 degrees, beyond the sliding limit
    assert alpha_large > alpha_sl, "Test assumption failed: alpha_large should be beyond sliding limit"
    
    Fy_large = calculate_front_tire_lateral_force(alpha_large, mu, load_f, C_alpha)
    expected_Fy_large = -mu * load_f  # Maximum lateral force
    assert jnp.isclose(Fy_large, expected_Fy_large, rtol=0.1), \
        f"Expected Fy≈{expected_Fy_large} with large slip angle, got {Fy_large}"
    
    # Test with negative slip angle
    alpha_neg = -0.05
    Fy_neg = calculate_front_tire_lateral_force(alpha_neg, mu, load_f, C_alpha)
    # Should be opposite sign of Fy_small
    assert jnp.isclose(Fy_neg, -Fy_small, rtol=0.1), \
        f"Expected Fy≈{-Fy_small} with negative slip angle, got {Fy_neg}"


def test_front_tire_dynamics_batch_processing() -> None:
    """Test the batch processing capability for massively parallel simulation."""
    # Define a batch of slip angles from -π/2 to π/2
    batch_size = 1000
    slip_angles = jnp.linspace(-jnp.pi/2 + 0.01, jnp.pi/2 - 0.01, batch_size)
    
    # Constant parameters for all batch elements
    friction_coefficient = 1.0
    front_load = 1000.0
    cornering_stiffness = 1000.0
    
    # Test both class-based and legacy function approaches
    # Create an instance of the model
    model = FrontTireDynamics(cornering_stiffness=cornering_stiffness)
    
    # Create a vectorized version of both the class and function that operates on the first argument (slip angle)
    vectorized_legacy_model = vmap(calculate_front_tire_lateral_force, in_axes=(0, None, None, None))
    
    # For the class-based interface, we need to provide slip_ratio as well
    # Use the JIT-compatible method for vectorization
    vectorized_class_model = vmap(model.jit_call, in_axes=(0, 0, None, None))
    
    # Process all slip angles in a single operation - legacy function approach
    batch_forces = vectorized_legacy_model(
        slip_angles, 
        friction_coefficient, 
        front_load, 
        cornering_stiffness
    )
    
    # Process using class-based approach
    slip_ratios = jnp.zeros_like(slip_angles)  # Slip ratio is not used in this model
    batch_forces_cls = vectorized_class_model(
        slip_angles,
        slip_ratios,
        front_load,
        friction_coefficient
    )
    
    # Verify batch shape matches input shape for both approaches
    assert batch_forces.shape == slip_angles.shape, f"Expected shape {slip_angles.shape}, got {batch_forces.shape}"
    assert batch_forces_cls.shape == slip_angles.shape, f"Expected shape {slip_angles.shape}, got {batch_forces_cls.shape}"
    
    # Verify both approaches give the same results
    assert jnp.allclose(batch_forces, batch_forces_cls), "Legacy and class-based models produced different results"
    
    # Verify that forces at positive and negative angles have the expected symmetry
    midpoint = batch_size // 2
    positive_angles = slip_angles[midpoint:]
    negative_angles = slip_angles[:midpoint]
    positive_forces = batch_forces[midpoint:]
    negative_forces = batch_forces[:midpoint]
    
    # Forces should be antisymmetric: F(-α) = -F(α)
    for i in range(len(negative_angles)):
        neg_idx = midpoint - i - 1  # Index from the end of negative angles
        pos_idx = midpoint + i      # Matching positive angle index
        if pos_idx < len(positive_angles):
            assert jnp.isclose(negative_forces[neg_idx], -positive_forces[i], rtol=1e-5), \
                f"Force symmetry violated at angles {negative_angles[neg_idx]} and {positive_angles[i]}"


def test_front_tire_dynamics_extreme_angles() -> None:
    """Test front_tire_dynamics with slip angles beyond ±π/2."""
    mu = 1.0
    load_f = 1000.0
    C_alpha = 1000.0
    
    # Test with angle > π/2 (vehicle moving backwards)
    alpha_extreme = 2.0  # ~114.6 degrees, beyond π/2
    Fy_extreme = calculate_front_tire_lateral_force(alpha_extreme, mu, load_f, C_alpha)
    
    # According to the Matlab implementation, angles beyond π/2 are transformed
    # alpha = (pi-abs(alpha))*sign(alpha)
    # Which for positive alpha > π/2 becomes alpha = (π-alpha)
    expected_alpha_transformed = jnp.pi - alpha_extreme
    expected_Fy = -mu * load_f * jnp.sign(expected_alpha_transformed)
    
    assert jnp.isclose(Fy_extreme, expected_Fy, rtol=0.1), \
        f"Expected Fy≈{expected_Fy} with extreme slip angle, got {Fy_extreme}"

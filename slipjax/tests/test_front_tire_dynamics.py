"""Tests for front tire dynamics module."""
from typing import Any

import jax.numpy as jnp
import pytest

from slipjax.front_tire_dynamics import front_tire_dynamics


def test_front_tire_dynamics_sanity() -> None:
    """Verify that front_tire_dynamics produces expected outputs for basic inputs."""
    # Test parameters
    mu = 1.0  # Friction coefficient
    load_f = 1000.0  # Front tire load (N)
    C_alpha = 1000.0  # Cornering stiffness (N/rad)
    
    # Test with zero slip angle
    alpha_zero = 0.0
    Fy_zero = front_tire_dynamics(alpha_zero, mu, load_f, C_alpha)
    assert jnp.isclose(Fy_zero, 0.0), f"Expected Fy=0 with zero slip angle, got {Fy_zero}"
    
    # Test with small slip angle (in linear region)
    alpha_small = 0.05  # ~2.86 degrees
    Fy_small = front_tire_dynamics(alpha_small, mu, load_f, C_alpha)
    # In linear region, Fy ≈ -C_alpha * tan(alpha)
    expected_Fy_small = -C_alpha * jnp.tan(alpha_small)
    assert jnp.isclose(Fy_small, expected_Fy_small, rtol=0.1), \
        f"Expected Fy≈{expected_Fy_small} with small slip angle, got {Fy_small}"
    
    # Test with large slip angle (in saturation region)
    alpha_sl = jnp.arctan(3 * mu * load_f / C_alpha)
    alpha_large = 1.5  # ~86 degrees, beyond the sliding limit
    assert alpha_large > alpha_sl, "Test assumption failed: alpha_large should be beyond sliding limit"
    
    Fy_large = front_tire_dynamics(alpha_large, mu, load_f, C_alpha)
    expected_Fy_large = -mu * load_f  # Maximum lateral force
    assert jnp.isclose(Fy_large, expected_Fy_large, rtol=0.1), \
        f"Expected Fy≈{expected_Fy_large} with large slip angle, got {Fy_large}"
    
    # Test with negative slip angle
    alpha_neg = -0.05
    Fy_neg = front_tire_dynamics(alpha_neg, mu, load_f, C_alpha)
    # Should be opposite sign of Fy_small
    assert jnp.isclose(Fy_neg, -Fy_small, rtol=0.1), \
        f"Expected Fy≈{-Fy_small} with negative slip angle, got {Fy_neg}"


def test_front_tire_dynamics_extreme_angles() -> None:
    """Test front_tire_dynamics with slip angles beyond ±π/2."""
    mu = 1.0
    load_f = 1000.0
    C_alpha = 1000.0
    
    # Test with angle > π/2 (vehicle moving backwards)
    alpha_extreme = 2.0  # ~114.6 degrees, beyond π/2
    Fy_extreme = front_tire_dynamics(alpha_extreme, mu, load_f, C_alpha)
    
    # According to the Matlab implementation, angles beyond π/2 are transformed
    # alpha = (pi-abs(alpha))*sign(alpha)
    # Which for positive alpha > π/2 becomes alpha = (π-alpha)
    expected_alpha_transformed = jnp.pi - alpha_extreme
    expected_Fy = -mu * load_f * jnp.sign(expected_alpha_transformed)
    
    assert jnp.isclose(Fy_extreme, expected_Fy, rtol=0.1), \
        f"Expected Fy≈{expected_Fy} with extreme slip angle, got {Fy_extreme}"

"""Tests for the full vehicle dynamics model."""
import jax
import jax.numpy as jnp
import pytest
from jax import vmap

from slipjax.dynamics import VehicleParameters, calculate_vehicle_dynamics, wrap_to_pi


@pytest.fixture
def default_vehicle_params() -> VehicleParameters:
    """Create a default set of vehicle parameters for testing."""
    return VehicleParameters(
        front_axle_distance=1.0,        # 1m from CG to front axle
        rear_axle_distance=1.5,         # 1.5m from CG to rear axle
        mass=1500.0,                    # 1500kg vehicle mass
        moment_of_inertia=2500.0,       # 2500 kg*m^2 moment of inertia
        front_tire_load=5000.0,         # 5000N front tire load
        rear_tire_load=7000.0,          # 7000N rear tire load
        friction_coefficient=1.0,       # Friction coefficient
        sliding_friction_coefficient=0.8,  # Sliding friction coefficient
        cornering_stiffness=50000.0,    # Cornering stiffness
        longitudinal_stiffness=60000.0, # Longitudinal stiffness
        yaw_damping=0.02,               # Default yaw damping
        velocity_damping=0.025          # Default velocity damping
    )


def test_wrap_to_pi() -> None:
    """Test that the wrap_to_pi function correctly wraps angles to [-π, π]."""
    # Test cases: (input, expected_output)
    test_cases = [
        (0.0, 0.0),                    # No change
        (jnp.pi, jnp.pi),              # No change
        (-jnp.pi, -jnp.pi),            # No change
        (3 * jnp.pi, -jnp.pi),         # 3π -> -π
        (-3 * jnp.pi, jnp.pi),         # -3π -> π
        (jnp.pi / 2, jnp.pi / 2),      # No change
        (2 * jnp.pi + 0.5, 0.5),       # 2π + 0.5 -> 0.5
        (-2 * jnp.pi - 0.5, -0.5),     # -2π - 0.5 -> -0.5
    ]
    
    for input_angle, expected_output in test_cases:
        result = wrap_to_pi(input_angle)
        assert jnp.isclose(result, expected_output), \
            f"wrap_to_pi({input_angle}) = {result}, expected {expected_output}"


def test_stationary_vehicle(default_vehicle_params: VehicleParameters) -> None:
    """Test the dynamics when the vehicle is stationary."""
    # Initial state: stationary vehicle at origin
    state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x, y, yaw, vx, vy, yaw_rate
    
    # No control input
    control_input = jnp.array([0.0, 0.0])  # wheel_speed, steering_angle
    
    # Calculate dynamics
    state_derivatives = calculate_vehicle_dynamics(
        state, 
        control_input, 
        default_vehicle_params
    )
    
    # Verify all derivatives are zero when stationary with no input
    for i, derivative in enumerate(state_derivatives):
        assert jnp.isclose(derivative, 0.0), \
            f"Expected zero derivative for state[{i}], got {derivative}"


def test_straight_acceleration(default_vehicle_params: VehicleParameters) -> None:
    """Test dynamics during straight-line acceleration."""
    # Initial state: stationary vehicle at origin, pointing in +x direction
    state = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # Small initial velocity
    
    # Accelerate with no steering
    control_input = jnp.array([5.0, 0.0])  # wheel_speed > vx, no steering
    
    # Calculate dynamics
    state_derivatives = calculate_vehicle_dynamics(
        state, 
        control_input, 
        default_vehicle_params
    )
    
    # Expectations during straight-line acceleration:
    # 1. Position derivatives should match velocity
    assert jnp.isclose(state_derivatives[0], state[3]), "x-position derivative should match vx"
    assert jnp.isclose(state_derivatives[1], 0.0), "y-position derivative should be zero"
    
    # 2. No rotational motion
    assert jnp.isclose(state_derivatives[2], 0.0), "Yaw rate should be zero"
    assert jnp.isclose(state_derivatives[5], 0.0), "Yaw acceleration should be zero"
    
    # 3. Positive longitudinal acceleration, no lateral acceleration
    assert state_derivatives[3] > 0, "Longitudinal acceleration should be positive"
    assert jnp.isclose(state_derivatives[4], 0.0, atol=1e-5), "Lateral acceleration should be zero"


def test_cornering_dynamics(default_vehicle_params: VehicleParameters) -> None:
    """Test dynamics during cornering with steering input."""
    # Initial state: vehicle moving in +x direction at 10 m/s
    state = jnp.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    
    # Apply steering to the right
    steering_angle = 0.1  # ~5.7 degrees to the right
    control_input = jnp.array([10.0, steering_angle])  # Maintain speed, steer right
    
    # Calculate dynamics
    state_derivatives = calculate_vehicle_dynamics(
        state, 
        control_input, 
        default_vehicle_params
    )
    
    # Expectations during right turn:
    # 1. Positive yaw acceleration (turning right/clockwise)
    assert state_derivatives[5] > 0, "Yaw acceleration should be positive for right turn"
    
    # 2. Should develop lateral velocity (negative vy for right turn in body frame)
    assert state_derivatives[4] < 0, "Lateral acceleration should be negative for right turn"
    
    # 3. Try the other direction (left turn)
    control_input_left = jnp.array([10.0, -steering_angle])
    state_derivatives_left = calculate_vehicle_dynamics(
        state, 
        control_input_left, 
        default_vehicle_params
    )
    
    # Turning left should have opposite signs for yaw acceleration and lateral acceleration
    assert state_derivatives_left[5] < 0, "Yaw acceleration should be negative for left turn"
    assert state_derivatives_left[4] > 0, "Lateral acceleration should be positive for left turn"


def test_drift_tendencies(default_vehicle_params: VehicleParameters) -> None:
    """Test dynamics in conditions conducive to drifting."""
    # Initial state: vehicle moving with some lateral velocity (sliding)
    state = jnp.array([0.0, 0.0, 0.0, 10.0, 2.0, 0.2])  # includes lateral velocity and yaw rate
    
    # Counter-steering (opposite to yaw rate direction)
    steering_angle = -0.1  # Counter-steer
    control_input = jnp.array([12.0, steering_angle])  # Accelerate slightly while counter-steering
    
    # Calculate dynamics
    state_derivatives = calculate_vehicle_dynamics(
        state, 
        control_input, 
        default_vehicle_params
    )
    
    # For drifting, we expect to maintain or increase yaw rate despite counter-steering
    # This is a key characteristic of sustained drifting
    assert state_derivatives[2] >= 0, "Yaw rate should maintain despite counter-steering during drift"
    
    # Should still be significant lateral acceleration for sustained sliding
    assert jnp.abs(state_derivatives[4]) > 0.1, "Should maintain significant lateral acceleration during drift"


def test_obstacle_state_handling(default_vehicle_params: VehicleParameters) -> None:
    """Test that the dynamics properly handle obstacle states if present."""
    # State with obstacle positions
    state_with_obstacle = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10.0, 5.0])
    control_input = jnp.array([5.0, 0.0])
    
    # Custom obstacle velocity
    obstacle_velocity = jnp.array([2.0, 3.0])
    
    # Calculate dynamics with explicit obstacle velocity
    state_derivatives = calculate_vehicle_dynamics(
        state_with_obstacle, 
        control_input, 
        default_vehicle_params,
        obstacle_velocity=obstacle_velocity
    )
    
    # Obstacle derivatives should match the provided velocities
    assert jnp.isclose(state_derivatives[6], obstacle_velocity[0]), \
        "Obstacle x velocity should match provided value"
    assert jnp.isclose(state_derivatives[7], obstacle_velocity[1]), \
        "Obstacle y velocity should match provided value"
    
    # Test default obstacle velocity (when not provided)
    state_derivatives_default = calculate_vehicle_dynamics(
        state_with_obstacle, 
        control_input, 
        default_vehicle_params
    )
    
    # Default obstacle velocities should be [0, 1] per the original MATLAB code
    assert jnp.isclose(state_derivatives_default[6], 0.0), \
        "Default obstacle x velocity should be 0"
    assert jnp.isclose(state_derivatives_default[7], 1.0), \
        "Default obstacle y velocity should be 1"


def test_batch_processing(default_vehicle_params: VehicleParameters) -> None:
    """Test batch processing of dynamics for multiple states at once."""
    # Create a batch of 100 states with different initial velocities
    batch_size = 100
    velocities = jnp.linspace(1.0, 20.0, batch_size)
    
    # Base state template
    base_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Create batch by replacing longitudinal velocity
    def create_state(velocity):
        state = base_state.at[3].set(velocity)
        return state
    
    states = jax.vmap(create_state)(velocities)
    
    # Fixed control input for all states
    control_input = jnp.array([10.0, 0.1])  # Same wheel speed and steering for all
    
    # Vectorized dynamics calculation over the first argument (state)
    vectorized_dynamics = vmap(
        calculate_vehicle_dynamics, 
        in_axes=(0, None, None, None)
    )
    
    # Calculate dynamics for all states at once
    batch_derivatives = vectorized_dynamics(
        states, 
        control_input, 
        default_vehicle_params,
        None
    )
    
    # Batch shape should match input shape
    assert batch_derivatives.shape[0] == batch_size, \
        f"Expected {batch_size} results, got {batch_derivatives.shape[0]}"
    
    # Each result should have 6 state derivatives
    assert batch_derivatives.shape[1] == 6, \
        f"Expected 6 derivatives per state, got {batch_derivatives.shape[1]}"
    
    # Verify monotonicity in certain outputs based on increasing velocity
    # e.g., higher speeds should generally result in higher yaw accelerations for a fixed steering angle
    yaw_accelerations = batch_derivatives[:, 5]  # Extract all yaw accelerations
    
    # Check if yaw accelerations generally increase with velocity (allowing for some non-monotonicity)
    # We just check that the highest velocities produce higher yaw accelerations than lowest velocities
    assert jnp.mean(yaw_accelerations[-10:]) > jnp.mean(yaw_accelerations[:10]), \
        "Expected higher yaw accelerations at higher velocities"

    # Spot check a mid-range derivative calculation

"""Example of simulating and plotting multiple vehicle trajectories.

This script demonstrates how to simulate the vehicle dynamics model for a batch
of initial states and plot their trajectories using Matplotlib.
"""

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap
from jax.lax import scan

# Assuming the slipjax package is installed or accessible in the path
from slipjax.dynamics import VehicleParameters, calculate_vehicle_dynamics

# Define default vehicle parameters (replace with actual desired values)
default_vehicle_params = VehicleParameters(
    front_axle_distance=1.1,  # Example value
    rear_axle_distance=1.4,   # Example value
    mass=1500.0,
    moment_of_inertia=2200.0, # Example value
    front_tire_load=7500.0,   # Example value
    rear_tire_load=7500.0,    # Example value
    friction_coefficient=0.9, # Example value
    sliding_friction_coefficient=0.7, # Example value
    cornering_stiffness=55000.0, # Example value
    longitudinal_stiffness=65000.0, # Example value
    yaw_damping=0.02,
    velocity_damping=0.025
)

# Simulation parameters
dt = 0.01  # Time step (s)
simulation_time = 5.0  # Total simulation time (s)
num_steps = int(simulation_time / dt)
num_trajectories = 100

# Base state with a constant initial forward velocity of 10 m/s
base_state = jnp.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])  # x, y, yaw, vx, vy, yaw_rate

# Vary steering angles from -0.2 to 0.2 radians (about -11.5 to 11.5 degrees)
steering_angles = jnp.linspace(-0.2, 0.2, num_trajectories)

# Fixed wheel speed for all simulations
wheel_speed = 10.0  # Example target wheel speed (rad/s)

# JIT compile the dynamics function
dynamics_jit = jit(calculate_vehicle_dynamics, static_argnums=(2,))

# Define the simulation step function using Euler integration
def simulation_step(state, steering_angle):
    """Performs one step of the simulation."""
    # Create control input from wheel speed and steering angle
    control_input = jnp.array([wheel_speed, steering_angle])
    
    # Calculate derivatives and update state with Euler integration
    state_derivative = dynamics_jit(state, control_input, default_vehicle_params)
    next_state = state + state_derivative * dt
    return next_state

# Function to simulate a complete trajectory for one steering angle
def simulate_trajectory(initial_state, steering_angle):
    """Simulates a complete trajectory with the given steering angle."""
    # Initialize array to store all states in the trajectory
    trajectory = jnp.zeros((num_steps + 1, 6))  # +1 to include initial state
    trajectory = trajectory.at[0].set(initial_state)
    
    # Define the loop body for fori_loop
    def body_fun(i, traj):
        # Get current state and calculate next state
        current_state = traj[i]
        next_state = simulation_step(current_state, steering_angle)
        # Store the next state in the trajectory array
        return traj.at[i+1].set(next_state)
    
    # Run the simulation steps using JAX's fori_loop (more efficient than Python loop)
    trajectory = jax.lax.fori_loop(0, num_steps, body_fun, trajectory)
    
    return trajectory

# Vectorize the trajectory simulation function to handle many steering angles at once
batch_simulator = vmap(simulate_trajectory, in_axes=(None, 0))

# Run the simulation for all steering angles
all_trajectories = batch_simulator(base_state, steering_angles)

# Create directory for saving the plot if it doesn't exist
os.makedirs('examples', exist_ok=True)

# Plot the trajectories - use subset to avoid overcrowding
plt.figure(figsize=(12, 10))

# Plot every Nth trajectory to avoid overcrowding
plot_step = 5  # Adjust this value to show more or fewer trajectories
for i in range(0, num_trajectories, plot_step):
    # Get the position coordinates from the trajectory
    trajectory = all_trajectories[i]
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]
    
    # Plot with color gradient based on steering angle
    # Use alpha for better visibility when trajectories overlap
    plt.plot(x_coords, y_coords, 
             label=f'Steering: {steering_angles[i]:.2f} rad',
             alpha=0.7)

# Add plot styling
plt.title('Vehicle Trajectories with Varying Steering Angles', fontsize=16)
plt.xlabel('X Position (m)', fontsize=14)
plt.ylabel('Y Position (m)', fontsize=14)
plt.grid(True)
plt.axis('equal')

# Place legend outside of the plot area to avoid covering the trajectories
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

# Save the plot with high resolution
plt.savefig("slipjax/examples/steering_trajectory_plot.png", bbox_inches='tight', dpi=300)
plt.close()

print("Simulation complete. Plot saved to slipjax/examples/steering_trajectory_plot.png")

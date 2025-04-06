"""Example of simulating and plotting multiple vehicle trajectories.

This script demonstrates how to simulate the vehicle dynamics model for a batch
of initial states and plot their trajectories using Matplotlib.
"""

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
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

# Function to draw the vehicle as a body with 4 wheels
def draw_vehicle(ax, x, y, yaw, steering_angle, vehicle_params, color='blue'):
    """Draw the vehicle as a body rectangle with 4 wheels at each corner.
    
    Args:
        ax: Matplotlib axis to draw on
        x, y: Position of the vehicle center in global coordinates
        yaw: Yaw angle of the vehicle
        steering_angle: Steering angle of the front wheels
        vehicle_params: Vehicle parameters including dimensions
        color: Color of the vehicle outline
    """
    # Vehicle dimensions
    body_length = vehicle_params.front_axle_distance + vehicle_params.rear_axle_distance  # No buffer
    body_width = 1.4  # Narrower car width
    wheel_length = 1.25  # Larger wheel length
    wheel_width = 0.6  # Larger wheel width
    track_width = 1.4  # Slightly wider track for wheel positioning
    
    # Calculate coordinates for the vehicle body rectangle in local frame
    body_points = np.array([
        [-vehicle_params.rear_axle_distance, -body_width/2],  # rear left
        [-vehicle_params.rear_axle_distance, body_width/2],   # rear right
        [vehicle_params.front_axle_distance, body_width/2],   # front right
        [vehicle_params.front_axle_distance, -body_width/2],  # front left
        [-vehicle_params.rear_axle_distance, -body_width/2]   # back to first point to close the shape
    ])
    
    # Calculate coordinates for a single wheel in local frame
    wheel_points = np.array([
        [-wheel_length/2, -wheel_width/2],
        [-wheel_length/2, wheel_width/2],
        [wheel_length/2, wheel_width/2],
        [wheel_length/2, -wheel_width/2],
        [-wheel_length/2, -wheel_width/2]
    ])
    
    # Transformation matrix for the vehicle body
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # Transform body points to global frame
    transformed_body_points = np.zeros_like(body_points)
    for i, point in enumerate(body_points):
        # Rotate
        rotated_x = point[0] * cos_yaw - point[1] * sin_yaw
        rotated_y = point[0] * sin_yaw + point[1] * cos_yaw
        # Translate
        transformed_body_points[i] = [x + rotated_x, y + rotated_y]
    
    # Calculate wheel positions in local vehicle coordinates
    wheel_positions = [
        # Front left wheel
        [vehicle_params.front_axle_distance, -track_width/2],
        # Front right wheel
        [vehicle_params.front_axle_distance, track_width/2],
        # Rear left wheel
        [-vehicle_params.rear_axle_distance, -track_width/2],
        # Rear right wheel
        [-vehicle_params.rear_axle_distance, track_width/2]
    ]
    
    # Draw the vehicle body
    ax.plot(transformed_body_points[:, 0], transformed_body_points[:, 1], color=color, linewidth=1.5)
    
    # Draw each wheel
    for i, wheel_pos in enumerate(wheel_positions):
        # Transform wheel position to global coordinates
        wheel_global_x = x + wheel_pos[0] * cos_yaw - wheel_pos[1] * sin_yaw
        wheel_global_y = y + wheel_pos[0] * sin_yaw + wheel_pos[1] * cos_yaw
        
        # Apply steering angle for front wheels (index 0 and 1)
        if i < 2:  # Front wheels
            wheel_angle = yaw + steering_angle
        else:  # Rear wheels
            wheel_angle = yaw
            
        cos_wheel = np.cos(wheel_angle)
        sin_wheel = np.sin(wheel_angle)
        
        # Transform wheel points to global frame
        transformed_wheel_points = np.zeros_like(wheel_points)
        for j, point in enumerate(wheel_points):
            # Rotate by the wheel angle
            rotated_x = point[0] * cos_wheel - point[1] * sin_wheel
            rotated_y = point[0] * sin_wheel + point[1] * cos_wheel
            # Translate to wheel position
            transformed_wheel_points[j] = [wheel_global_x + rotated_x, wheel_global_y + rotated_y]
        
        # Draw the wheel
        ax.plot(transformed_wheel_points[:, 0], transformed_wheel_points[:, 1], color='black', linewidth=1.5)

# Plot the trajectories - use subset to avoid overcrowding
fig, ax = plt.subplots(figsize=(12, 10))

# Plot every Nth trajectory to avoid overcrowding
plot_step = 5  # Adjust this value to show more or fewer trajectories
for i in range(0, num_trajectories, plot_step):
    # Get the position coordinates from the trajectory
    trajectory = all_trajectories[i]
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]
    yaw = trajectory[:, 2]
    
    # Plot with color gradient based on steering angle
    # Use alpha for better visibility when trajectories overlap
    ax.plot(x_coords, y_coords, 
           label=f'Steering: {steering_angles[i]:.2f} rad',
           alpha=0.7)
    
    # Draw the vehicle at specific points along the trajectory
    draw_step = num_steps // 4  # Draw vehicle at fewer points to avoid clutter
    for j in range(0, num_steps, draw_step):
        draw_vehicle(ax, 
                    x_coords[j], 
                    y_coords[j], 
                    yaw[j], 
                    steering_angles[i], 
                    default_vehicle_params, 
                    color='blue')

# Add plot styling
ax.set_title('Vehicle Trajectories with Varying Steering Angles', fontsize=16)
ax.set_xlabel('X Position (m)', fontsize=14)
ax.set_ylabel('Y Position (m)', fontsize=14)
ax.grid(True)
ax.set_aspect('equal')

# Place legend outside of the plot area to avoid covering the trajectories
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

# Save the plot with high resolution
fig.savefig("examples/steering_trajectory_plot.png", bbox_inches='tight', dpi=300)
plt.close(fig)

print("Simulation complete. Plot saved to examples/steering_trajectory_plot.png")

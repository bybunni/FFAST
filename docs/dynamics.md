# Vehicle Dynamics Model Documentation

## dynamics.m

The `dynamics.m` file models the drifting dynamics of a vehicle based on the paper "Dynamics And Control Of Drifting In Automobiles" by Hindiyeh (2013). The function `dynamics` computes the derivative of the state vector `X` given the current state, control inputs, and optional parameters.

### Inputs
- **X**: State vector `[x; y; psi; v_x; v_y; psi_dot]`
- **U**: Control input `[wheel_speed; steer_angle]`
- **params**: Optional parameters, including obstacle velocity

### Outputs
- **X_dot**: Derivative of the state vector

### Global Variables
- **vehicle**: Contains vehicle parameters such as lengths, masses, and inertias
- **debug**: Stores debugging information

### Key Components
- **Tire Dynamics**: Uses `tire_dyn_f` and `tire_dyn_r` to calculate lateral and longitudinal forces
- **Vehicle Dynamics**: Computes the rate of change of the vehicle's state, considering both inertial and damping effects

## tire_dyn_f.m

The `tire_dyn_f.m` file models the front tire dynamics. It calculates the lateral force `F_y` based on the slip angle `alpha`.

### Inputs
- **alpha**: Slip angle

### Outputs
- **Fy**: Lateral force

### Global Variables
- **vehicle**: Contains parameters like cornering stiffness and load

## tire_dyn_r.m

The `tire_dyn_r.m` file models the rear tire dynamics. It calculates both longitudinal and lateral forces `F_x` and `F_y` based on the slip angle `alpha` and wheel slip `K`.

### Inputs
- **v_x**: Longitudinal velocity
- **wheel_vx**: Wheel longitudinal velocity
- **alpha**: Slip angle

### Outputs
- **Fx**: Longitudinal force
- **Fy**: Lateral force

### Global Variables
- **vehicle**: Contains parameters like cornering stiffness, load, and friction coefficients
- **debug**: Stores debugging information

These files collectively model the vehicle dynamics necessary for simulating drifting behavior.

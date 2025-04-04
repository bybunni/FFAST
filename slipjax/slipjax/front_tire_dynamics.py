import jax.numpy as jnp
from jax import jit
import jax.lax as lax


@jit
def front_tire_dynamics(alpha, mu, load_f, C_alpha):
    """
    Calculate the lateral force Fy for the front tire given the slip angle alpha.

    Parameters:
    alpha (float): Slip angle in radians.
    mu (float): Friction coefficient.
    load_f (float): Load on the front tire.
    C_alpha (float): Cornering stiffness.

    Returns:
    float: Lateral force Fy.
    """
    # Handle extreme slip angles (> pi/2)
    alpha = lax.cond(
        jnp.abs(alpha) > jnp.pi / 2,
        lambda a: (jnp.pi - jnp.abs(a)) * jnp.sign(a),
        lambda a: a,
        alpha
    )

    alpha_sl = jnp.arctan(3 * mu * load_f / C_alpha)
    
    # Calculate Fy based on slip angle magnitude
    def calc_linear_region(a):
        return (
            -C_alpha * jnp.tan(a)
            + C_alpha**2 / (3 * mu * load_f) * jnp.abs(jnp.tan(a)) * jnp.tan(a)
            - C_alpha**3 / (27 * mu**2 * load_f**2) * jnp.tan(a) ** 3
        )
    
    def calc_saturation_region(a):
        return -mu * load_f * jnp.sign(a)
    
    Fy = lax.cond(
        jnp.abs(alpha) <= alpha_sl,
        calc_linear_region,
        calc_saturation_region,
        alpha
    )

    return Fy

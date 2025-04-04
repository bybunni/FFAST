import jax.numpy as jnp
from jax import jit


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
    if jnp.abs(alpha) > jnp.pi / 2:
        alpha = (jnp.pi - jnp.abs(alpha)) * jnp.sign(alpha)

    alpha_sl = jnp.arctan(3 * mu * load_f / C_alpha)
    if jnp.abs(alpha) <= alpha_sl:
        Fy = (
            -C_alpha * jnp.tan(alpha)
            + C_alpha**2 / (3 * mu * load_f) * jnp.abs(jnp.tan(alpha)) * jnp.tan(alpha)
            - C_alpha**3 / (27 * mu**2 * load_f**2) * jnp.tan(alpha) ** 3
        )
    else:
        Fy = -mu * load_f * jnp.sign(alpha)

    return Fy

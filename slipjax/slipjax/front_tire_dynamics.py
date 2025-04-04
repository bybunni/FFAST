"""Front tire dynamics model.

This module contains functions for calculating the front tire dynamics based on
the model described in 'Dynamics And Control Of Drifting In Automobiles' 
(Hindiyeh, 2013).
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit
import jax.lax as lax


@jit
def front_tire_dynamics(
    alpha: jax.Array, 
    mu: jax.Array, 
    load_f: jax.Array, 
    C_alpha: jax.Array
) -> jax.Array:
    """Calculate the lateral force Fy for the front tire given the slip angle alpha.

    Args:
        alpha: Slip angle in radians.
        mu: Friction coefficient.
        load_f: Load on the front tire in Newtons.
        C_alpha: Cornering stiffness in N/rad.

    Returns:
        Lateral force Fy in Newtons.
    """
    # Handle extreme slip angles (> pi/2)
    alpha = lax.cond(
        jnp.abs(alpha) > jnp.pi / 2,
        lambda a: (jnp.pi - jnp.abs(a)) * jnp.sign(a),
        lambda a: a,
        alpha,
    )

    alpha_sl = jnp.arctan(3 * mu * load_f / C_alpha)
    
    # Calculate Fy based on slip angle magnitude
    def calc_linear_region(a: jax.Array) -> jax.Array:
        """Calculate lateral force in the linear (non-saturated) region."""
        return (
            -C_alpha * jnp.tan(a)
            + C_alpha**2 / (3 * mu * load_f) * jnp.abs(jnp.tan(a)) * jnp.tan(a)
            - C_alpha**3 / (27 * mu**2 * load_f**2) * jnp.tan(a) ** 3
        )
    
    def calc_saturation_region(a: jax.Array) -> jax.Array:
        """Calculate lateral force in the saturation region."""
        return -mu * load_f * jnp.sign(a)
    
    Fy = lax.cond(
        jnp.abs(alpha) <= alpha_sl,
        calc_linear_region,
        calc_saturation_region,
        alpha,
    )

    return Fy

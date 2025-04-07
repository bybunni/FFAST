"""Helper functions for working with JAX in the SlipJAX package."""

from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
from jax import lax

T = TypeVar('T')


def safe_division(
    numerator: jax.Array, denominator: jax.Array, fallback_value: jax.Array = 0.0
) -> jax.Array:
    """Safely divide arrays, handling division by zero.
    
    Args:
        numerator: The numerator in the division
        denominator: The denominator in the division
        fallback_value: Value to use when denominator is zero (default: 0.0)
        
    Returns:
        Result of safe division operation
    """
    return lax.cond(
        jnp.abs(denominator) > 1e-10,
        lambda x: numerator / x,
        lambda _: fallback_value,
        denominator,
    )


def jit_compatible_switch(
    condition: jax.Array, true_fn: Callable[..., T], false_fn: Callable[..., T], *args: Any
) -> T:
    """A wrapper around lax.cond for readability in if/else conditions.
    
    This is a more readable alternative to lax.cond for JIT-compatible conditional execution.
    
    Args:
        condition: Boolean condition to evaluate
        true_fn: Function to call if condition is True
        false_fn: Function to call if condition is False
        *args: Arguments to pass to the selected function
        
    Returns:
        Result from either true_fn or false_fn
    """
    return lax.cond(condition, true_fn, false_fn, *args)

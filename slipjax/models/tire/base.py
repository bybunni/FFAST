"""Base classes and interfaces for tire dynamics models."""

from abc import ABC, abstractmethod
from typing import Protocol

import jax
import jax.numpy as jnp


class TireDynamicsModel(Protocol):
    """Protocol defining the interface for tire dynamics models."""
    
    def __call__(
        self,
        slip_angle: jax.Array,
        slip_ratio: jax.Array,
        load: jax.Array,
        friction_coefficient: jax.Array,
    ) -> jax.Array:
        """Calculate tire lateral force.
        
        Args:
            slip_angle: The slip angle of the tire in radians
            slip_ratio: The slip ratio of the tire (dimensionless)
            load: The normal load on the tire in Newtons
            friction_coefficient: The friction coefficient between tire and road
            
        Returns:
            The calculated lateral force in Newtons
        """
        ...


class BaseTireDynamics(ABC):
    """Base class for tire dynamics models."""
    
    @abstractmethod
    def __call__(
        self,
        slip_angle: jax.Array,
        slip_ratio: jax.Array,
        load: jax.Array,
        friction_coefficient: jax.Array,
    ) -> jax.Array:
        """Calculate tire lateral force.
        
        Args:
            slip_angle: The slip angle of the tire in radians
            slip_ratio: The slip ratio of the tire (dimensionless)
            load: The normal load on the tire in Newtons
            friction_coefficient: The friction coefficient between tire and road
            
        Returns:
            The calculated lateral force in Newtons
        """
        pass

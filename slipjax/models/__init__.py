"""Models for vehicle and tire dynamics.

This package contains the various dynamics models used in SlipJAX,
including tire models and vehicle dynamics simulations.
"""

# Import all submodules to make them available via slipjax.models
from slipjax.models import tire
from slipjax.models import vehicle

__all__ = [
    'tire',
    'vehicle'
]

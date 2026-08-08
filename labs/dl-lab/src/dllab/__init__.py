"""dllab: utilidades reutilizables del laboratorio de deep learning."""

from dllab.utils.device import describe_device, get_device
from dllab.utils.seed import set_seed

__all__ = ["get_device", "describe_device", "set_seed"]
__version__ = "0.1.0"

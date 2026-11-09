"""Optional common controller interface, outside the V1 prediction benchmark."""
from .cem import CEMController, Controller

__all__ = ["CEMController", "Controller"]

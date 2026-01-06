"""An artificial intelligence library built from scratch using pure Python and Numpy."""

from . import backend
from . import feature_extraction
from . import nn
from . import preprocessing
from . import supervised

__all__ = [
    'backend',
    'feature_extraction',
    'nn',
    'preprocessing',
    'supervised',
]
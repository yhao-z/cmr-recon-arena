"""Array-backend abstraction: ``ndarray`` is a numpy-or-cupy type alias and
``get_ndarray_module`` returns the module (np/cp) that owns an array.
"""

from typing import TypeAlias

import numpy as np
try:
    import cupy as cp
    xp = cp
except ImportError:  # importing cupy fails if no cuda is available
    cp = None
    xp = np


ndarray: TypeAlias = np.ndarray | xp.ndarray  # type: ignore


def get_ndarray_module(array: ndarray):
    """Return numpy or cupy depending on which module owns `array`."""
    if isinstance(array, np.ndarray):
        return np
    elif cp is not None and isinstance(array, cp.ndarray):
        return cp
    else:
        raise TypeError(f'Array has invalid type {type(array)}')

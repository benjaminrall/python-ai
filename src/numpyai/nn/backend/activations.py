"""Neural network activation functions."""

import numpy as np
from numpy.typing import NDArray

def leaky_relu(x: NDArray, alpha: float = 0.3) -> NDArray:
    """Leaky rectified linear unit activation function, `leaky_relu(x) = max(x, alpha * x)`.

    Parameters
    ----------
    alpha : float, optional
        A float that controls the slope for values lower than 0. Defaults to 0.3.
    """
    return np.maximum(x, alpha * x)

def relu(x: NDArray) -> NDArray:
    """Rectified linear unit activation function, `relu(x) = max(x, 0)`."""
    return np.maximum(x, 0)

def sigmoid(x: NDArray) -> NDArray:
    """Sigmoid activation function, `sigmoid(x) = 1 / (1 + exp(-x))`.
    
    For small values `sigmoid` returns a value close to 0 and 
    for large values it returns a value close to 1.
    """
    return 1.0 / (1.0 + np.exp(-x))

def stable_sigmoid(x: NDArray) -> NDArray:
    """Numerically stable sigmoid activation function.

    This method avoids overflow by calculating exponentials 
    for negative and positive values separately.
    """
    # Calculates exponentials for negative and positive values separately
    neg_exp = np.exp(x[x < 0])
    pos_exp = np.exp(-x[x >= 0])

    # Creates output array using separate equations for stability
    z = np.zeros(x.shape)
    z[x < 0] = neg_exp / (1.0 + neg_exp)
    z[x >= 0] = 1.0 / (1.0 + pos_exp)
    return z

def softmax(x: NDArray) -> NDArray:
    """Softmax activation function.

    Converts vectors of values to probability distributions.
    The elements of the output vector are in the range [0, 1] and sum to 1. 

    Each vector in the input is handled independently. The softmax of each
    vector x is computed as `softmax(x) = exp(x) / reduce_sum(exp(x))`.
    """
    # Ensures numerical stability by subtracting the maximum value from the input
    max_val = np.max(x, axis=-1, keepdims=True)

    # Calculates exponentials and their sums
    exps = np.exp(x - max_val)
    sums = np.sum(exps, axis=-1, keepdims=True)

    # Returns final probabilities array
    return exps / sums

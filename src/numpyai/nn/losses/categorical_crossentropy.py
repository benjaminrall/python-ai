"""Categorical cross-entropy loss function class."""

from numpy.typing import NDArray
from numpyai.nn.backend import categorical_crossentropy, normalise_output
from .loss import Loss

class CategoricalCrossentropy(Loss):
    """
    Computes the categorical cross-entropy loss between the outputs and targets.
    
    This cross-entropy loss is used when there are two or more output 
    classes where target outputs are provided in a one-hot representation.
    """

    identifier = 'categorical_crossentropy'
    aliases = ['cce']

    def __init__(self, from_logits: bool = False) -> None:
        """Computes the categorical cross-entropy loss between the outputs and targets.
        
        This cross-entropy loss is used when there are two or more output 
        classes where target outputs are provided in a one-hot representation.

        Parameters
        ----------
        from_logits : bool, optional
            Whether the predicted outputs are represented as 
            logits or as probabilities.
        """
        self.from_logits = from_logits

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the result of the categorical cross-entropy loss function."""
        return categorical_crossentropy(output, target, self.from_logits)

    def derivative(self, output: NDArray, target: NDArray) -> NDArray:
        """Calculates the derivative of the categorical cross-entropy loss function."""
        return normalise_output(output, self.from_logits) - target

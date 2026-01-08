"""Base loss function class."""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from numpyai.backend import Registrable, Representable

class Loss(Representable, Registrable['Loss'], ABC):
    """Abstract base class from which all neural network loss functions inherit."""

    identifier: str
    """The loss function's string identifier."""
    
    def __call__(self, output: NDArray, target: NDArray) -> float:
        """
        Calculates the result of the loss function for a 
        given set of outputs and targets.
        """
        return self.call(output, target)

    @abstractmethod
    def call(self, output: NDArray, target: NDArray) -> float:
        """
        Calculates the result of the loss function for a 
        given set of outputs and targets.
        """

    @abstractmethod
    def derivative(self, output: NDArray, target: NDArray) -> NDArray:
        """
        Calculates the derivatives of the loss function with 
        respect to each of the network's outputs.
        """

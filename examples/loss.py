import numpy as np
import math
from abc import ABC, abstractmethod


class Loss(ABC):
  def __init__(self):
    super().__init__()

  def calculate(self, output_layer, y):
    """
    Calculate the mean value of the sample losses.
    """
    sample_losses = self.forward(output_layer, y)
    data_loss = np.mean(sample_losses)

    return data_loss

  @abstractmethod
  def forward(self, batch, y):
    """
    Compute the loss value
    """
    raise NotImplementedError(
        "Please implement the forward method for the class.")

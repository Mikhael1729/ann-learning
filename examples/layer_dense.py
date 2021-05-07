import numpy as np


class DenseLayer:
  def __init__(self, input_size, size):
    """
    Initialize the layer.

    size: Neurons quantity for the new layer.
    input_size: The neurons quantity of the previuos layer.
    """
    # The `size * input_size` dimension allows to skip a transpose operation in forward.
    self.weights = 0.10 * np.random.randn(input_size, size)
    # Bias for each neuron in the new layer.
    self.biases = np.zeros((1, size))

  def forward(self, batch):
    """
    Generates the output of the layer with the given batch of inputs.

    Returns a matrix with the 
    """
    self.output = np.dot(np.array(batch), self.weights) + self.biases


import numpy as np

np.random.seed(0)


class DenseLayer:
  def __init__(self, size, input_size):
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
    self.output = np.dot(batch, self.weights) + self.biases


# Batch sample data. The rows Indicates the 
batch = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

layer1 = DenseLayer(5, 4) # Layer of 
layer2 = DenseLayer(2, 5)

layer1.forward(batch)
layer2.forward(layer1.output)

print(layer1.output)
print(layer2.output)

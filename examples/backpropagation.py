import numpy as np


def forward_process_example():
  """
  ANN with 2 layers: input (with 3 neurons) output (with 1 neuron)
  """
  X = np.array([1.0, -2.0, 3.0])  # Values of input neurons
  W = np.array([-3.0, -1.0, 2.0])  # Weights to the output neuron
  b = 1.0  # Bias of the output neuron.

  input_multiplication = X * W  # [x0w0, x1w1, x2w2]
  neuron_sum = np.sum(input_multiplication) + b  # z
  neuron_activation = max(neuron_sum, 0)  # y

  return neuron_activation, neuron_sum, input_multiplication, X, W, b


def backpropagation_example():
  """

  In the backpropagation process you have as parameters the weights and biases of the
  connections.

  1. (value) For each output neuron (for this case just one), compute the ReLU derivative with
     respect to the sum... (its parameter)
  2. (array) The partial derivative of the sum with respect to each member.
  3. (array) The partial derivative of each multiplication with respect to each parameter.
  """
  # Derivative from a hypothetical next neuron.
  next_neuron_value = 1.0

  # Values from the forward process.
  y, z, M, X, W, b = forward_process_example()

  # 1. Derivative of the ReLU
  z_derivative = 1.0 if z > 0 else 0.0

  # 2. Partial derivative of the sum with respect to each member (n1, n2, n3, b)
  sum_partials = np.array([1, 1, 1, 1])

  # 3. Partial derivatives of multiplications to respect to each parameter.
  multiplication_partials = np.array(
      [[W[0], X[0]], [W[1], X[1]], [W[2], X[2]]])

  # Apply the chain rule. Has the relu derivatives for each neuron in the network.
  relu1 = z_derivative * next_neuron_value
  relu2 = sum_partials * relu1
  relu3 = (multiplication_partials.T * relu2[:-1]).T

  relu_partials = [relu1, relu2, relu3]
  for relu in relu_partials:
    print(relu)

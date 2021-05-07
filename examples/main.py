"""
Here I test the code examples
"""
import nnfs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from layer_dense import DenseLayer
from activation_relu import ActivationRelu
from softmax import softmax, numpy_softmax
from activation_softmax import ActivationSoftmax
from nnfs.datasets import spiral_data
from loss import Loss
from loss_categorical_crossentropy import CategoricalCrossEntropy
from backpropagation import backpropagation_example

# Allows to matplolib use an interface to show the plots.
matplotlib.use('Qt5Agg')

# It configures some elements of numpy.
nnfs.init()


def test_DenseData():
  """
  Tests the DenseData class.
  """
  # Batch sample data. The rows Indicates the
  batch = [
      [1, 2, 3, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8]
  ]

  layer1 = DenseLayer(4, 5)
  layer2 = DenseLayer(5, 2)

  layer1.forward(batch)
  # Using the previuos layer as input of this new one.
  layer2.forward(layer1.output)

  print(layer1.output)
  print(layer2.output)


def test_ActivationRelu():
  """
  Tests the ActivationRelu class.
  """
  batch, y = spiral_data(100, 3)  # 100 feature sets of 3 classes.

  layer = DenseLayer(2, 5)
  layer.forward(batch)

  activation = ActivationRelu()
  activation.forward(layer.output)  # The negative values are now gone.

  # Print the modified output.
  print(activation.output)

  # Print the data.
  plt.scatter(batch[:, 0], batch[:, 1], c=y, cmap='brg')
  plt.show()


def test_softmax():
  """
  Tests the softmax function.
  """
  output_layer = [4.8, 1.21, 2.385]
  normalized_layer = softmax(output_layer)
  normalized_numpy_layer = numpy_softmax(output_layer)

  # Original
  print(output_layer)

  # Printing the result of my two versions of the softmax function.
  print(list(normalized_layer))
  print(normalized_numpy_layer)


def test_softmax_and_layers():
  """
  Creates a two layer neural network using the ReLU activation function for the hidden layer
  and the Softmax activation function for the output one.

  Returns the output of the 
  """
  # Data.
  batch, y = spiral_data(100, 3)  # 100 feature sets of 3 classes.

  # Creating layers.
  # The value 2 is because spiral data has 2 variables. The 3 is arbitrary.
  dense1 = DenseLayer(2, 3)
  activation1 = ActivationRelu()

  # Because the output of the previuos layer, the first parameter is a 3.
  # Because the data has 3 clases, the second parameter is a 3.
  dense2 = DenseLayer(3, 3)
  activation2 = ActivationSoftmax()

  dense1.forward(batch)
  activation1.forward(dense1.output)

  dense2.forward(activation1.output)
  activation2.forward(dense2.output)

  # Print the first 5 of the 300 results.
  print(activation2.output[:5])

  return activation2.output, y


def test_loss():
  softmax_outputs = np.array([[0.7, 0.1, 0.2],
                              [0.1, 0.5, 0.4],
                              [0.02, 0.9, 0.08]])

  # Batch of one-hot encoded labels. Each one indicates the correct answer with 1.
  class_targets = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0]])

  categorical_cross_entropy = CategoricalCrossEntropy()
  loss = categorical_cross_entropy.calculate(softmax_outputs, class_targets)
  print(loss)


def test_loss_with_softmax():
  output, y = test_softmax_and_layers()
  categorical_cross_entropy = CategoricalCrossEntropy()
  loss = categorical_cross_entropy.calculate(output, y)

  print(loss)


def test_accuaracy_with_softmax():
  output, y = test_softmax_and_layers()
  predictions = np.argmax(output, axis=1)

  if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

  accuracy = np.mean(predictions == y)

  print(accuracy)

if __name__ == "__main__":
  # test_DenseData()
  # test_ActivationRelu()
  # test_softmax()
  # test_softmax_and_layers()
  # test_loss()
  # test_loss_with_softmax()
  # test_accuaracy_with_softmax()
  backpropagation_example()



import math
import numpy as np


def softmax(layer):
  numerators = list(map(lambda value: math.e**value, layer))
  denominator = sum(numerators)
  normalized_layer = map(lambda numerator: numerator / denominator, numerators)
  return normalized_layer

def numpy_softmax(layer):
  numerators = np.exp(layer)
  normalized_layer = numerators / np.sum(numerators)

  return normalized_layer

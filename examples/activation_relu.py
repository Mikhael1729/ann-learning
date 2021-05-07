import numpy as np

class ActivationRelu:
  def forward(self, inputs):
    self.output = np.maximum(0, inputs)

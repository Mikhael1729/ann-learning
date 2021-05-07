import numpy as np


class ActivationSoftmax:
  """
  ActivationSoftmax implements the softmax activation function in the forward method
  of the class.
  """

  def forward(self, batch):
    """
    Applies the softmax activation function to all neruon values in the batch.

    Parameters:

      batch: A batch of output neurons.

    Return another batch with the same dimensions but with the output neuron values normalized.
    """
    # Because e^n where n is a number between -inf and 0 only results on values between
    # 0 and 1, you can use it to bound your results in that range.
    # Convert each value of each sample of the batch as follows:
    # - First, gets the max value on each sample of the batch
    # - Substract it from each value from the original batch
    # - Apply the np.exp to the resulting value.
    exp_values = np.exp(batch - np.max(batch, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    self.output = probabilities

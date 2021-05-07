
from loss import Loss
import numpy as np


class CategoricalCrossEntropy(Loss):
  def forward(self, batch, target_classes):
    # Clip data to prevent division by 0.
    y_pred_clipped = np.clip(batch, 1e-7, 1 - 1e-7)

    if len(target_classes.shape) == 1:
      correct_confidences = y_pred_clipped[range(len(batch)), target_classes]
    elif len(target_classes.shape) == 2:
      correct_confidences = np.sum(y_pred_clipped * target_classes, axis=1)

    negative_log_loss = -np.log(correct_confidences)
    return negative_log_loss

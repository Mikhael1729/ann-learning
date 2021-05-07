# Categorical Cross-Entropy Loss Function

It's necessary a method for determining how wrong is the model. In general, The loss function for choice (at least classifications using softmax as the activation function on the output layer) is the Cross-Entropy Loss.

## Mathematical notation

The following is the mathematical representation for this loss function:

$$L_i=-\sum{y_{i,j}\ log\left(\hat{y}_{i,\ j}\right)} = -log\left(\hat{y}_{i,\ k}\right)$$

- $L$ is the sample loss value
- Where $i$ is a sample index on a batch
- $j$ a value in the sample into a batch.
- $\hat{y}$ the predicted values (the output of the neuron or the predicted distribution)
- $y$ the target values (the correct output or the actual/disired distribution)
- $k$ is the index of _true_ probability.

## Mathematical computation

Imagine you have a model with the following elements:

| Element                           | Value             |
| --------------------------------- | ----------------- |
| Classes                           | 3                 |
| Label (index of true probability) | 0                 |
| One-hot                           | `[1, 0, 0]`       |
| Prediction                        | `[0.7, 0.1, 0.2]` |

> **One hot**: Is an array where only a value is "hot" (on), with a value 1, and the rest are "cold" (off), with values of 0.

The computation of this function looks like this:

$$
\begin{aligned}
L_i&=-\sum{y_{j}\ log\left(\hat{y}_{j}\right)}\\
&=-\left(1 \cdot log\left(0.7\right) + 0 \cdot log\left(0.1 \right) + 0 \cdot log\left(0.2 \right)\right)\\
&=-\left(-0.35667494393873245 + 0 + 0\right) \\
&= 0.35667494393873245
\end{aligned} \\
$$

## Python computation

This is the basic operation using raw Python:

```python
import math

# Values of the output layer of the ANN.
softmax_output = [0.7, 0.1, 0.2]

# Ground truth
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss) # Output: 0.35667494393873245
```

This is the operation using batches:

```python
import numpy as np
"""
Compute loss using numpy and batches.

In the example I represent the class targets as a batch of class targets and using the
categorical labels representation.
"""
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# Batch of one-hot encoded labels. Each one indicates the correct answer with 1.
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

class_targets_type = len(class_targets.shape)
# Categorical label indices (each value indicates the index of the true value)
# class_targets = np.array([0, 1, 1]) # This is the same as the above but in other notation.

if class_targets_type == 1:
  sample_indices = range(len(softmax_outputs))
  correct_confidences = softmax_outputs[sample_indices, class_targets]
elif class_targets_type == 2:
  # Compute the confidence for each sample. axis = 1 indicates that wil sum the rows.
  correct_confidences = np.sum(softmax_outputs * class_targets, axis=1)

neg_log = -np.log(correct_confidences)
average_loss = np.mean(neg_log)

print(average_loss)
```

## Loss interpretation

The larger the value in the correct class, the lost is lower. When the confidence is lower in the correct class, the higher the loss becomes

## HMM

### Using a batch of outputs

| Element              | Value                                                   |
| -------------------- | ------------------------------------------------------- |
| Classes              | 3 (Dog, Cat, Human)                                     |
| Softmax outputs      | `[[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]` |
| Class target indices | [0, 1, 1] dog, cat, cat                                 |


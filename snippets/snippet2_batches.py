import numpy as np

# 3 input layer samples.
batches = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
]

# 4 Connections (columns) to each output neuron (3 rows)
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# 3 Biases (for output neurons).
biases = [2, 3, 0.5]

outputs = np.dot(batches, np.array(weights).T) + biases

print(outputs)

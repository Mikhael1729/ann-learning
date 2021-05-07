# Accuaracy

Describes how often the largest confidence is the correct class in terms of a fraction

## Visualization

```
# Samples (with correct class at index 1)
s1 = [0.8, 0.0, 0.2]
s2 = [0.2, 0.5, 0.3]
s3 = [0.6, 0.2, 0.2]

softmax_outputs = np.array([s1, s2, s3])

# Is expected to have the higher values on the indicated indices.
class_targets = [0, 1, 1]

# Is the actual result.
predictions = [0, 1, 0]

# 2 / 3 = 0.666
accuracy = np.mean(predictions == class_targets) 
```

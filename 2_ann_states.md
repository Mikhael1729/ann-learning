# Artificial Neural Network States

There re two main pervasive challenges with neural networks: 

- Dead nerons. Is the term used to a ANN that has 0 as value for weights and biases in so many neurons that then produces no response to the given input.
- Exploiding values. Are computed values (for example, the process of an activation function) that can produce enormous values (tending to infinity)

## Ways to avoid them

There are some ways to avoid this ANN escenarios:

## Avoid dead neurons

Dead neurons and enormouse numbers, can wreak havoc down the line in render a network useless over time. To avoid this situation, initialize the weights and biases with numbers different to zero. An example is to initialize them is using values of the normal distribution.

## Avoid epxploiding values

An example of how to avoid them can be seen on my implementation of the softmax activation function.

One of the steps to generate the value of the softmax activation function is to raise $e$ to the $n$ where $n$ is a given input layer value. A small number in the operation like $n=1000$ can raise an overflow exception in the language.

Thanks to the normalization softmax does one can make the result of the operation between 0 and 1 have the same result (which is a nice property to exploit). To make the result of the operation between 0 and 1, do the following:

- Get the max value on each value of the inputs.
- Substract it on each value from the original inputs
- Compute the result with the obtained value of the previous step:

    $n = e^{v - m}$

  Where $v$ is a given input value, $m$ the maximum value obtained in the inputs.


This is a nice pseudocde visualization of the process:
```
# Inputs
I = [-0.23, 0.54, 4, 45]

# The max value of the inputs array is 45
max = 45

# Substract it on each input.
S = [(-0.23 - 45), (0.54 - 45), (4 - 45), (45 - 45)]

# The results
S = [-45.23, -44.46, -41, 0]

N = numpy.exp(S) # Contains values between 0 and 1
```

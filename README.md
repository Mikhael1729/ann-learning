# Artificial Neural Networks (ANN)

An ANN is a function that given certain input can give you the answer for a given problem. It's composed of artificial neurons that act like little calculators of tiny parts to control the fitting on a given set of data. Each one contributes to the overall result in the output process.

Here is a condesed resume of my learning on basic neural networks concepts.

## Artificial Neuron

It's a mathematical function that takes as arguments _inputs_ $X=\{x_1,x_2,...,x_n\}$, _weights_ of the inputs $W=\{w_1,w_2,...,w_n\}$, a _bias_ $b$ (a value that when is greater makes the neuron more easily to fire the higher value in the output range), and returns a value called $output$ which is between a discrete (usually two) or continous range of values.

> To **fire** a neuron means to update its output value (process the function)

$$f(X, W, b) = output$$

The different models of artificial neurons are the same in structure but differ on firing and output range or possible values. The most known artificial neurons are:

- Perceptrons
- Sigmoids.

### Perceptron neuron

Fires a 0 if the sum of the inputs and weights plus the bias is greather than 0, otherwise a 1.

$$f(X,W,b)=\Big\{^{0\ if\ \sum{x_i w_i}\ +\ b\ >\ 0}_{1\ if \sum{x_i w_i}\ +\ b\  \leq \ 0}$$

### Sigmoid neuron

Fires any number from 0 to 1. This kind of values allows you to have a small change in the output from a small change in the input. This differs from the perceptron neuron where a small variation in the input or weights of the neuron produces unexpected and very probably big differences in the output.

The output of the sigmoid neuron can be a function called _sigmoid function_ that looks like this:

$$\sigma(z)=\frac{1}{1+e^{-z}}$$

And applied to artificial neurons:

$$\sigma\left(\sum w_i x_i+b\right)=\frac{1}{1+e^{-(\sum w_i x_i -b)}}$$

## Linear unite

It's fast (requires less computation than sigmoid). It's very used in hhidden layers.

$$f(X,W,b)=\Big\{^{x\ if\ \sum{x_i w_i}\ +\ b\ >\ 0}_{1\ if \sum{x_i w_i}\ +\ b\  \leq \ 0}$$

## Neural network training

The process of fine-tuning the weights and biases from the input data is known as _traning the neural network_.

It is a process of three steps:

1. Calculate the predicted output $\hat{y}$, known as **feedforward**.
2. Choose a **loss function** and use it to calculate the error between each predicted value.
3. Update the weights and biases, known as **backpropagation**.

### Feedforward process

Feedorward is the firing process of the network. Consists on computing the value of each output neuron (which imples to calculate the output value of the connected neurons).

**Example**. Mathematically you can represent the feedforward process of a two layer network as follows:

$$
\hat{y} = \sigma \left(W_2 \cdot \sigma \left(W_1 \cdot X + b_1\right) + b \right)
$$

> Here I'm using the _dot_ notation which represents the sum of the weights $W$ and values $X$.

### Loss function

// TODO:

## Weights and biases

The tuning to weights and biases defines the learning process for a NN. This is the straight line equation:

$$y = mx + b$$

You can represent the output of a given neuron like the straight line equation:

$$output = input \cdot weight + bias$$

## Activation functions

- Perceptron
- Sigmoid. This has an issue referred to the vanishing gradient problem
- Rectified Linear Unit. It's fast (less computation than sigmoid). It's very used in hidden layers.
- Softmax. It's very used on classifier model. It's apparently used in the output layers of ReLU hidden layers.

## References

- [How to build your own neural network](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)


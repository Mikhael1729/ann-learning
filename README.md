# Artificial Neural Networks (ANN)

An ANN is a function that given certain input can give you the answer for a given problem. It's composed of artificial neurons that act like little calculators of tiny parts to control the fitting on a given set of data. Each one contributes to the overall result in the desired output.

## Artificial Neuron

An artificial neuron is a function that takes inputs $X$ and weights $W$ as parameters, computes the sum of the product between each input $x_i$ and weight $w_i$ plus a bias $b$ and uses an activation function $g$ that depends on the previous sum to produce a value that can be used by other neurons in an ANN:

$$
f(X, W) = g\left(\sum{x_iw_i} + \ b\right)
$$

There are many artificial neurons and they only differ on the activation function they use. Some of the most commonly used artificial neurons are: perceptron neuron and sigmoid neuron.

## Activation functions

Activation functions are mathematical operations to tweak the behavior (output) of a certain artificial neuron. There are several of them, the most commonly used (of the ones I know) are the Step Function, the Sigmoid Function, the Rectified Linear Unit (ReLU) and Softmax.

If the network of an ANN contains neurons with linear activation functions the resulting model only can shape linear data. On the other hand, using non-linear activation functions like the Sigmoid or ReLU ones, the model can shape the data by using the correct quantity of layers and the correct amount of neurons on each one

> To represent non linear data an ANN needs at least two hidden layers. Remember each layer can only change a section of the resulting model. That's the reason why you need more than one.

### Rectified Linear Unit (ReLU)

It's the most commonly used activation function and mainly used on hidden layers of an ANN. The reason is more used than the sigmoid function is because it requires small computation to get a result (less than the sigmoid function).

Mathematical representation:

$$ReLU(x)=\Big\{^{x\ \ if\ x >\ 0}_{0\ \  if x \ \leq \ 0}$$

### Binary step

Fires a 0 if the sum of the inputs and weights plus the bias is greather than 0, otherwise a 1.

$$f(x)=\Big\{^{0\ if\ z +\ b\ >\ 0}_{1\ if \sum{x_i w_i}\ +\ b\  \leq \ 0}$$

### Sigmoid Function

Fires any number from 0 to 1. This kind of values allows you to have a small change in the output from a small change in the input. This differs from the step function where a small variation in the input or weights produces unexpected and very probably big differences in the output.

The output of the sigmoid neuron can be a function called _sigmoid function_ that looks like this:

$$\sigma(x)=\frac{1}{1+e^{-x}}$$

### Softmax function

It's normally used for neurons in the output layer of an ANN to normalize the output of a network to a probability distribution over predicted output classes.

$$
S(Y)=\frac{e^{y_i}}{\sum_{j=1}^{n} e^{y_{j}}}
$$

Where $Y$ is the list of the neuron values of a given layer, $y_i$ and $y_j$ each element on the list and $n$ the quantity of values in the list

## Neural network training

The process of fine-tuning the weights and biases from the input data is known as _traning the neural network_.

It is a process of three steps:

1. Calculate the predicted output $\hat{y}$, known as **feedforward**.
2. Choose a **loss function** and use it to calculate the error between each predicted value.
3. Update the weights and biases, known as **backpropagation**.

### Feedforward propagation

The feedforward propagation is the firing process of the network. Consists on computing the value of each output neuron (which imples to calculate the output value of the connected neurons).

**Example**. Mathematically you can represent the feedforward process of a two layer network as follows:

$$
\hat{y} = \sigma \left(W_2 \cdot \sigma \left(W_1 \cdot X + b_1\right) + b \right)
$$

> Here I'm using the _dot_ notation which represents the sum of the weights $W$ and values $X$.

$$output = input \cdot weight + bias$$

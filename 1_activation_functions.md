# Activation functions

Activation functions are mathematical operations to tweak the behavior (output) of a certain artificial neuron. There are several of them, the most commonly used (of the ones I know) are the Step Function, the Sigmoid Function, the Rectified Linear Unit (ReLU), Softmax.

If the network of an ANN contains neurons with linear activation functions the resulting model only can shape linear data. On the other hand, using non-linear activation functions like the Sigmoid or ReLU ones, the model can shape the data with the correct quantity of layers and the correct amount of neurons on each one.

> To represent non linear data an ANN needs at least two hidden layers. Remember each layer can only change a section of the resulting model. That's the reason why you need more than one.

## Rectified Linear Unit (ReLU)

It's the most commonly used activation function and mainly used on hidden layers of an ANN. The reason is more used than the sigmoid function is because it requires small computation to get a result (less than the sigmoid function).

Mathematical representation:

$$f(x)=\Big\{^{x\ \ if\ x >\ 0}_{0\ \  if x \ \leq \ 0}$$


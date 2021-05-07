import matplotlib.pyplot as plt
import numpy as np


def f(x): return 2*x


X = np.array(range(5))
Y = f(X)

print("X: ", X)
print("Y: ", Y)

plt.plot(X, Y)
plt.show()

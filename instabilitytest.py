import numpy as np

def sigmoid_stable(x):
    "Numerically-stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)

def sigmoid_instable(x):
    return 1/(1+np.exp(-x))

print(sigmoid_instable(100000))
print(sigmoid_instable(-100000))

print(sigmoid_stable(-100000))
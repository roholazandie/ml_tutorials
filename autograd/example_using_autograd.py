import autograd as ag
import autograd.numpy as np

def tanh(x):
    if x<1:
        y = np.exp(-x)
    else:
        y = np.exp(x)
    return (1.0-y)/(1.0+y)

tanh_grad = ag.grad(tanh)
print(tanh_grad(1.0))

print((tanh(1.000001) - tanh(0.999999))/0.000002)

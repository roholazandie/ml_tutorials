import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin, fmin_ncg, brent
import numpy as npp
from scipy.optimize.optimize import golden, fmin_cg

'''
If we define the problem of learning as an optimization problem 
then we can find out there are multiple ways to find the minimum of
lost (or risk) function as a function of parameters.
In this script I tried to find the best parameters using different methods
I also plot the training_loss function
'''


def training_loss(params):
    W, b = params
    linear_model = W * x + b
    squared_delta = np.square(linear_model - y)
    return np.sum(squared_delta)


x = np.array([1, 2, 3, 4])
y = np.array([0,-1,-2,-3])

w = b = npp.arange(-10.0, 10.0, 0.05)
W, B = npp.meshgrid(w, b)
zs = npp.array([training_loss(npp.array([x,y])) for x,y in zip(npp.ravel(W), npp.ravel(B))])
Z = zs.reshape(W.shape)

import plots
plots.surface(W, B, Z)




# Optimize weights using downhill simplex algorithm
min_params = fmin(training_loss, npp.array([0, 0]))
print(min_params)
print(training_loss(min_params))

# Optimize weights using conjugate gradient method.
min_params = fmin_cg(training_loss, npp.array([0, 0]))
print(min_params)
print(training_loss(min_params))

# Optimize weights using gradient descent.
training_gradient_fun = grad(training_loss)

weights = np.array([0.0, 1.0])
print("Initial loss:", training_loss(weights))
for i in range(10000):
    weights -= training_gradient_fun(weights) * 0.01

print(weights)
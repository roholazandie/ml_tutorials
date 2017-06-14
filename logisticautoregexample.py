import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin

def sigmoid(x):
    #return 0.5*(np.tanh(x) + 1)
    return 1./(1+np.exp(-x))

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    #preds = logistic_predictions(weights, inputs)
    #label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    #return -np.sum(np.log(np.abs(label_probabilities)))

    preds = logistic_predictions(weights, inputs)
    label_probabilities = targets * np.log(preds) + (1-targets)*(1-np.log(preds))
    return -np.sum(label_probabilities)

def f(x):   # The rosenbrock function
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

# Build a toy dataset.
inputs = np.array([[0.52, 1.12],
                   [0.88, -1.08],
                   [0.52, 0.06],
                   [0.74, -2.49]])
targets = np.array([1, 1, 0, 1])

# Define a function that returns gradients of training loss using autograd.
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.
weights = np.array([0.0, 1.0])
print("Initial loss:", training_loss(weights))
for i in range(1000):
    weights -= training_gradient_fun(weights) * 0.01
    #print(weights)
print(weights)
print ("Trained loss:", training_loss(weights))


min_f = fmin(training_loss, np.array([0,0]))
print(min_f)
print(training_loss(min_f))
###################################################
import numpy as npp


x = y = npp.arange(-5.0, 14.0, 0.05)
X, Y = npp.meshgrid(x, y)
zs = npp.array([f(npp.array([x,y])) for x,y in zip(npp.ravel(X), npp.ravel(Y))])
Z = zs.reshape(X.shape)

import plots
plots.surface(X, Y, Z)
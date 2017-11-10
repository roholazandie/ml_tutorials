import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
x = np.linspace(-2, 2, 100)
for i in range(10):
    df = nd.Derivative(np.tanh, n=i)
    y = df(x)
    h = plt.plot(x, y/np.abs(y).max())
#plt.show()

xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
ydata = 1+2*np.exp(0.75*xdata)
fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
Jfun = nd.Jacobian(fun)
np.allclose(np.abs(Jfun([1,2,0.75])), 0)



##################################################
def affine(W, X, b):
    return np.dot(W, X) + b

def sigmoid(X):
    return 1. / (1 + np.exp(-X))

def squared(X):
    return X**2

def derivative_squred(X):
    return 2*X

def derivative_sigmoid(X):
    return sigmoid(X)*(1-sigmoid(X))

def loss(X):
    return np.linalg.norm(X)**2


X = np.array([[3.0], [2.0]])
W = np.array([[2.0, 1.0], [1.0, 2.0]])
b = np.array([[1.0], [2.0]])
Y = np.dot(W, X) + b
A = squared(Y)
T = np.array([[0.0], [1.0]])

grad_L_A = nd.Gradient(loss)(A-T)
jacobian_A_Y = nd.Jacobian(squared)(Y)

#grad_L_Y = np.dot(jacobian_A_Y.T, grad_L_A)
grad_L_Y = np.dot(jacobian_A_Y.T, grad_L_A)
#grad_L_Y = jacobian_A_Y.T * grad_L_A
print(grad_L_Y)

affine = lambda w: np.dot(X.T, w) + b.T
jacobian_Y_W = nd.Jacobian(affine)(W)
grad_L_W = np.dot(jacobian_Y_W[1:,].T, grad_L_Y[np.newaxis,:])
#grad_L_W = jacobian_Y_W * grad_L_Y
print(grad_L_W)

affine = lambda b: np.dot(X.T, W) + b.T
jacobian_Y_b = nd.Jacobian(affine)(b)
print(jacobian_Y_b)

grad_L_b = np.dot(jacobian_Y_b.T, grad_L_Y)
print(grad_L_b)
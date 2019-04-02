import autograd.numpy as np
from autograd import grad
from autograd.convenience_wrappers import elementwise_grad
from autograd import jacobian
from autograd.util import flatten


def affine(W, X, b):
    f, _ = flatten(np.dot(W, X) + b)
    return f

def sigmoid(X):
    return 1. / (1 + np.exp(-X))

def squared(X):
    return X**2

def derivative_squred(X):
    return 2*X

def derivative_sigmoid(X):
    return sigmoid(X)*(1-sigmoid(X))

def loss(Y, T):
    return np.linalg.norm(Y - T) ** 2

def objective(params):
    X = np.array([[3.0], [2.0]])
    W = params[0]
    b = params[1]
    Y = affine(W, X, b).reshape((2, 1))

    #A = sigmoid(Y)
    A = squared(Y)

    T = np.array([[0.0], [1.0]])
    return loss(A, T)

if __name__ == "__main__":
    W1 = np.array([[2.0, 1.0], [1.0, 2.0]])
    b1 = np.array([[1.0], [2.0]])
    objective_grad = grad(objective)
    result = objective_grad([W1, b1])
    print(result)

    ###################################
    X = np.array([[3.0], [2.0]])
    W = W1
    b = b1
    Y = affine(W, X, b).reshape((2, 1))
    A = squared(Y)
    T = np.array([[0.0], [1.0]])


    grad_L_A = 2*(A-T)
    grad_A_Y = 2*Y
    grad_L_Y = grad_L_A * grad_A_Y
    print(grad_L_Y)

    grad_Y_W = X
    grad_L_W = np.dot(grad_L_Y, grad_Y_W.T)
    print(grad_L_W)

    ################################
    X = np.array([[3.0], [2.0]])
    W = W1
    b = b1
    Y = affine(W, X, b).reshape((2, 1))
    A = squared(Y)
    T = np.array([[0.0], [1.0]])

    grad_L_A = 2 * (A - T)
    grad_A_Y = 2 * Y
    grad_L_Y = grad_L_A * grad_A_Y
    print(grad_L_Y)

    grad_Y_W = X
    grad_L_W = np.dot(grad_L_Y, grad_Y_W.T)
    print(grad_L_W)
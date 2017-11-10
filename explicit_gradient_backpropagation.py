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


def feedforward(X, parameters):
    activations = []
    outputs = []
    for W, b in parameters:
        output = affine(W, X, b)
        activation = sigmoid(output)
        X = activation
        activations.append(activation)
        outputs.append(output)

def loss(A, Y):
    return np.linalg.norm(A - Y)**2

def objective(params):
    X = np.array([[3.0], [2.0]])
    W1 = params[0]
    b1 = params[2]
    Y1 = affine(W1, X, b1).reshape((2, 1))
    A1 = sigmoid(Y1)

    W2 = params[1]
    b2 = params[3]
    Y2 = affine(W2, A1, b2).reshape((2, 1))
    A2 = sigmoid(Y2)

    T = np.array([[1.5], [1.0]])
    return loss(A2,T)

if __name__ == "__main__":


    #################################################
    X = np.array([[3.0], [2.0]])
    W1 = np.array([[2.0, 1.5], [0.5, 2.5]])
    b1 = np.array([[1.0], [2.0]])
    Y1 = affine(W1, X, b1).reshape((2, 1))
    A1 = sigmoid(Y1)

    W2 = np.array([[2.0, 1.5], [0.5, 2.5]])
    b2 = np.array([[1.0], [6.3]])
    Y2 = affine(W2, A1, b2).reshape((2, 1))
    A2 = sigmoid(Y2)

    T = np.array([[1.5], [1.0]])
    ##################################################
    objective_grad = grad(objective)
    result = objective_grad([W1, W2, b1, b2])
    print(result)
    ##################################################

    grad_L_A2 = grad(loss, 0)
    grad_L_A2_result = grad_L_A2(A2, T)
    #print(grad_L_A2_result)

    Y2 = np.array([[1.0], [2.0]])
    sigmoid_derivative = elementwise_grad(sigmoid)
    sigmoid_derivative_Y2 = sigmoid_derivative(Y2)
    grad_L_Y2 = np.multiply(grad_L_A2_result, sigmoid_derivative_Y2)
    #print(grad_L_Y2)

    A1 = np.array([[2.0], [1.0]])
    A1, _ = flatten(A1)
    b2, _ = flatten(b2)

    jacobian_Y2_A1 = jacobian(affine, 1)
    jacobian_Y2_A1_result = jacobian_Y2_A1(W2, A1, b2)
    #print(jacobian_Y2_A1_result.shape)
    #print(jacobian_Y2_A1_result)

    grad_L_A1_result = np.dot(jacobian_Y2_A1_result, grad_L_Y2)
    #print(grad_L_A1_result)

    Y1 = np.array([[1.0], [2.0]])
    sigmoid_derivative = elementwise_grad(sigmoid)
    sigmoid_derivative_Y1 = sigmoid_derivative(Y1)
    grad_L_Y1 = np.multiply(grad_L_A1_result, sigmoid_derivative_Y1)
    #print(grad_L_Y1)


    grad_L_W2 = np.dot(grad_L_Y2, A1.reshape((2, 1)).T)
    grad_L_b2 = np.multiply(grad_L_Y2, b2.reshape((2, 1)))

    grad_L_W1 = np.dot(grad_L_Y1, X.T)
    grad_L_b1 = np.multiply(grad_L_Y1, b1)

    print(grad_L_W1)
    print(grad_L_b2)
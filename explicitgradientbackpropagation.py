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

if __name__ == "__main__":
    X = np.array([[3.0], [2.0]])
    W = np.random.randn(2, 2)
    b = np.random.randn(2, 1)
    # print(W.T)
    # print(b)
    # print(X)
    # print(affine(W, X, b))
    # print(sigmoid(affine(W, X, b)))
    #################################################
    A2 = np.array([[1.0], [2.0]])
    T = np.array([[1.5], [1.0]])
    grad_L_A2 = grad(loss, 0)
    grad_L_A2_result = grad_L_A2(A2, T)
    print(grad_L_A2_result)

    Y2 = np.array([[1.0], [2.0]])
    sigmoid_derivative = elementwise_grad(sigmoid)
    sigmoid_derivative_Y2 = sigmoid_derivative(Y2)
    grad_L_Y2 = np.multiply(grad_L_A2_result, sigmoid_derivative_Y2)
    print(grad_L_Y2)

    A1 = np.array([[2.0], [1.0]])
    A1, _ = flatten(A1)
    b2 = np.array([[1.0], [6.3]])
    b2, _ = flatten(b2)
    W2 = np.array([[2.0, 1.5], [0.5, 2.5]])

    affine_jacobian = jacobian(affine, 1)
    resultl_affine_jacobian = affine_jacobian(W2, A1, b2)
    print(resultl_affine_jacobian.shape)
    print(resultl_affine_jacobian)
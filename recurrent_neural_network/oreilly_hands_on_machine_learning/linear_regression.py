import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

def generate_dataset():
    X = 2*np.random.rand(100, 1)
    y = 4 + 3*X+ np.random.rand(100, 1)
    return X, y


def normal_equation(X, y):
    '''
    Solve linear regression using the normal equation(exact closed form solution)
    the problem with this is that it uses the whole dataset which is not efficient
    :param X:
    :return:
    '''
    X = np.hstack((np.ones((100, 1), dtype='float32'), X))
    theta_best = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
    return theta_best


def batch_gradient_descnet(X, y, etha, n_iteration):
    '''
    here instead we use batch gradient descnet
    which uses the gradient of minimum squared error(MSE) instead of solving it to get
    the closed form answer. then using the step method:
    x_(i+1) = x_i + stepsize * gradient(MSE)
    we solve the problem
    WARNING: when using gradient descent you should ensure that all
    features have the same scale or it takes much longer to converge
    :param X:
    :param etha:
    :param n_iteration:
    :return:
    '''
    X = np.hstack((np.ones((100, 1), dtype='float32'), X))
    n_samples, n_features = np.shape(X)
    theta = np.random.rand(n_features, 1)
    for i in range(n_iteration):
        theta = theta - etha * (2/n_samples)* X.T.dot(X.dot(theta)-y)

    return theta

def stochastic_gradient_descent(X, y):
    '''
    the problem with batch gradient descent is that it still uses the whole
    dataset in stochastic gadient descent we use just one random instance per iteration
    and then try this in multiple epochs.
    :param X:
    :param y:
    :return:
    '''
    X = np.hstack((np.ones((100, 1), dtype='float32'), X))
    n_samples, n_features = np.shape(X)
    m = 1000
    etha = 0.1
    gamma = 0.001
    n_epochs = 100
    theta = np.random.rand(n_features, 1)
    for _ in range(n_epochs):
        for _ in range(m):
            rand_index = np.random.randint(n_samples)
            X_random_sample = X[rand_index, :].reshape(1, n_features)
            y_random_sample = y[rand_index, :]
            theta = theta - etha * (2 / n_samples) * X_random_sample.T * (X_random_sample.dot(theta) - y_random_sample)
            #etha = etha * (1-gamma)
    return theta


def minibatch_stochastic_gradient_descent(X, y):
    '''
    the problem with stochastic gradient descent is that it uses
    only one instance per iteration that makes it unstable
    in mini batch we compromise between stochastic and batch solutions
    this increase the stability and speed because it can use GPU powered
    matrix multiplication for reasonable minibatch sizes
    :param X:
    :param y:
    :return:
    '''
    X = np.hstack((np.ones((100, 1), dtype='float32'), X))
    n_samples, n_features = np.shape(X)
    m = 1000
    etha = 0.1
    gamma = 0.001
    n_epochs = 100
    minibatch_size = 10
    theta = np.random.rand(n_features, 1)
    for _ in range(n_epochs):
        for _ in range(m):
            rand_index = np.random.randint(n_samples, size = (1, minibatch_size))
            X_random_sample = X[rand_index, :].reshape(minibatch_size, n_features)
            y_random_sample = y[rand_index, :].reshape(minibatch_size, 1)
            theta = theta - etha * (2 / n_samples) * X_random_sample.T.dot((X_random_sample.dot(theta) - y_random_sample))
            # etha = etha * (1-gamma)
    return theta


def stochastic_gradient_descent_using_sklearn(X, y):
    sgd_regressor = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
    sgd_regressor.fit(X, y.ravel())
    return sgd_regressor.intercept_, sgd_regressor.coef_

if __name__ == "__main__":
    X, y = generate_dataset()
    plt.scatter(X, y)

    theta_best = normal_equation(X, y)
    print(theta_best)

    theta_best = batch_gradient_descnet(X, y, etha=0.1, n_iteration=1000)
    print(theta_best)

    theta_best = stochastic_gradient_descent(X, y)
    print(theta_best)

    theta_best = minibatch_stochastic_gradient_descent(X, y)
    print(theta_best)

    intercept, coef = stochastic_gradient_descent_using_sklearn(X, y)
    print(intercept)
    print(coef)
    plt.plot(X, coef * X + intercept, color='r')

    #plt.plot(X, theta_best[0] * X + theta_best[1], color='r')
    plt.show()
import numpy as np
from sklearn import datasets
from sklearn.linear_model.logistic import LogisticRegression
import matplotlib.pyplot as plt
import plotlyvisualization as plotly
import math




def sigmoid(X):
    "Numerically-stable sigmoid function."
    sig_result = []
    for x in X:
        if x >= 0:
            z = math.exp(-x)
            sig_result.append(1 / (1 + z))
        else:
            z = math.exp(x)
            sig_result.append(z / (1 + z))

    return np.array(sig_result)


def load_dataset_unimodal():
    iris = datasets.load_iris()
    X = iris["data"][:, 3:]
    y = (iris["target"]==2).astype(np.int)
    return X, y


def load_dataset_multimodal():
    iris = datasets.load_iris()
    X = iris["data"][:, (1, 3)]
    y = iris["target"]
    return X, y


def logistic_regression(X, y):
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X, y)
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_prob = logistic_reg.predict_proba(X_new)
    #y_prob = np.array([sigmoid(logistic_reg.coef_ * X_new + logistic_reg.intercept_)])
    plt.plot(X_new, y_prob[:, 1], 'r', label="Iris")
    plt.plot(X_new, y_prob[:, 0], 'b--', label="Not Iris")
    plt.show()


def one_versus_all_logistic_regression(X, y):
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X, y)
    x1 = np.linspace(0, 10, 100)
    y1 = np.linspace(0, 10, 100)
    logistic_reg.predict([[5, 2]])
    X = np.zeros(shape=(100, 100))
    for i, iv in enumerate(x1):
        for j, jv in enumerate(y1):
            # X[i, j] = softmax_reg.predict([[iv, jv]])
            X[i, j] = max(logistic_reg.predict_proba([[iv, jv]])[0])

    plotly.plot_surface(X)


def multinomial_logistic_regression(X, y):
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
    softmax_reg.fit(X, y)
    x1 = np.linspace(0, 10, 100)
    y1 = np.linspace(0, 10, 100)
    softmax_reg.predict([[5, 2]])
    X = np.zeros(shape=(100, 100))
    for i, iv in enumerate(x1):
        for j, jv in enumerate(y1):
            #X[i, j] = softmax_reg.predict([[iv, jv]])
            X[i, j] = max(softmax_reg.predict_proba([[iv, jv]])[0])


    plotly.plot_surface(X)


def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

def logistic_regression_manual(X, ys):
    X = np.hstack((np.ones((150, 1), dtype='float32'), X))
    n_iteration = 20000
    etha = 5e-5
    n_samples, n_features = np.shape(X)
    theta = np.random.rand(n_features, 1)
    mm = []
    for i in range(n_iteration):
        grad = 0
        for x, y in zip(X, ys):
            score = np.dot(x, theta)
            predictions = sigmoid(score)
            output_error = y - predictions
            #grad += np.dot(x.T, output_error).reshape((n_features, 1))
            grad += output_error*x.T.reshape((n_features, 1))


        theta = theta + etha * grad
        print(theta)
        #print(log_likelihood(X, ys, theta))
        #y_pred = np.array([sigmoid(x.dot(theta)) for x in X])
        #losses = np.array([y[j]*np.log(y_pred[j]) + (1-y[j])*(1-np.log(y_pred[j])) for j in range(n_samples)])
        # loss = -np.sum(losses)
        # mm.append(loss)
        # print(theta)

    #theta = np.array([-4.22, 2.61]).reshape(2,1)
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_prob = np.array([sigmoid(theta[1]* X_new + theta[0])]).T

    # prob =  X_new.dot(theta)
    plt.plot(X_new, y_prob, 'r', label="Iris")
    plt.show()
    # #plotly.plot(mm)
    # print(theta)

    return theta


if __name__ == "__main__":
    X, y = load_dataset_unimodal()
    #one_versus_all_logistic_regression(X, y)
    #logistic_regression(X, y)

    #X, y = load_dataset_multimodal()
    #multinomial_logistic_regression(X, y)

    logistic_regression_manual(X, y)

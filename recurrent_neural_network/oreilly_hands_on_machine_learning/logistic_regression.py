import numpy as np
from sklearn import datasets
from sklearn.linear_model.logistic import LogisticRegression
import matplotlib.pyplot as plt
import plotlyvisualization as plotly


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
    plt.plot(X_new, y_prob[:, 1], 'r', label="Iris")
    plt.plot(X_new, y_prob[:, 0], 'b--', label="Not Iris")
    plt.show()


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

if __name__ == "__main__":
    X, y = load_dataset_unimodal()
    logistic_regression(X, y)

    X, y = load_dataset_multimodal()
    multinomial_logistic_regression(X, y)
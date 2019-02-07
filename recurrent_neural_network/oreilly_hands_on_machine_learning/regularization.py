import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.linear_model.stochastic_gradient import SGDClassifier, SGDRegressor
from sklearn.metrics.regression import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import clone


def generate_dataset():
    m = 100
    X = 6 * np.random.rand(m ,1) - 3
    y = 0.5 * X**2 + X +  2 * np.random.randn(m, 1)
    return X, y

def poly_regression_quadratic(X, y):
    polynomial_features = PolynomialFeatures(degree=2)
    X_poly = polynomial_features.fit_transform(X)
    linear_regression = LinearRegression()
    linear_regression.fit(X_poly, y)
    return linear_regression.intercept_, linear_regression.coef_


def ridge_regression(X, y, alpha):
    polynomial_features = PolynomialFeatures(degree=2)
    X_poly = polynomial_features.fit_transform(X)
    standard_scalar = StandardScaler()
    #X_scaled = standard_scalar.fit(X_poly)
    ridge_reg = Ridge(alpha=alpha, solver="cholesky")
    ridge_reg.fit(X_poly, y)
    return ridge_reg.intercept_, ridge_reg.coef_



def stochastic_gradient_descnet_with_l2_penalty(X, y):
    '''
    sgd with l2 penalty is actually ridge regression
    :param X:
    :param y:
    :return:
    '''
    polynomial_features = PolynomialFeatures(degree=2)
    X_poly = polynomial_features.fit_transform(X)
    sgd_regressor = SGDRegressor(penalty='l2')
    sgd_regressor.fit(X_poly, y)
    return sgd_regressor.intercept_, sgd_regressor.coef_


def lasso_regression(X, y, alpha):
    polynomial_features = PolynomialFeatures(degree=10)
    X_poly = polynomial_features.fit_transform(X)
    standard_scalar = StandardScaler()
    # X_scaled = standard_scalar.fit(X_poly)
    lasso_reg = Lasso(alpha=alpha)
    lasso_reg.fit(X_poly, y)
    return lasso_reg.intercept_, lasso_reg.coef_


def elasticnet_regression(X, y, alpha):
    polynomial_features = PolynomialFeatures(degree=10)
    X_poly = polynomial_features.fit_transform(X)
    standard_scalar = StandardScaler()
    # X_scaled = standard_scalar.fit(X_poly)
    elasticnet_reg = ElasticNet(alpha=alpha, l1_ratio=0.5)
    elasticnet_reg.fit(X_poly, y)
    return elasticnet_reg.intercept_, elasticnet_reg.coef_


def early_stopping_sgd(X, y):
    n_sample = y.shape[0]
    X_train, y_train = X[:int(n_sample*0.7)], y[:int(n_sample*0.7)]
    X_test, y_test = X[int(n_sample*0.7):], y[int(n_sample*0.7):]

    polynomial_features = PolynomialFeatures(degree=2)
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_test_poly = polynomial_features.fit_transform(X_test)

    sgd_regressor = SGDRegressor(max_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
    minimum_value_error = float("inf")
    best_epoch = 0
    best_model = None
    n_epochs = 1000
    errors = []
    for epoch in range(n_epochs):
        sgd_regressor.fit(X_train_poly, y_train)
        y_pred = sgd_regressor.predict(X_test_poly)
        mse = mean_squared_error(y_test.ravel(), y_pred.ravel())
        errors.append(mse)
        if mse < minimum_value_error:
            minimum_value_error = mse
            best_epoch = epoch
            best_model = clone(sgd_regressor)

    plt.plot(errors)
    plt.show()
    return best_epoch, best_model


def plot_polynomial(X, coefs, intercept, color):
    X = np.sort(X, axis=0)  # for plotting it needs to be sorted
    Y = 0
    for i in range(len(coefs)):
        Y += coefs[i] * X**i
    Y += intercept
    plt.plot(X, Y, color=color)



if __name__ == "__main__":
    X, y = generate_dataset()


    # plt.scatter(X, y)
    #
    # intercept, coef = poly_regression_quadratic(X, y)
    # print(intercept)
    # print(coef)

    # for alpha, color in zip([0.0, 1e-5, 0.1, 1, 10.0, 100.0], ['r', 'g', 'b', 'c', 'k', 'y']):
    #     intercept, coef = ridge_regression(X, y, alpha=alpha)
    #
    #     X = np.sort(X, axis=0) # for plotting it needs to be sorted
    #     plt.plot(X, coef[0][2] * X**2 + coef[0][1] * X + intercept, color=color)

    # intercept, coef = stochastic_gradient_descnet_with_l2_penalty(X, y)
    #
    # X = np.sort(X, axis=0)  # for plotting it needs to be sorted
    # plt.plot(X, coef[2] * X ** 2 + coef[1] * X + intercept, color='r')

    # for alpha, color in zip([0.0, 1e-5, 0.1, 1, 10.0, 100.0], ['r', 'g', 'b', 'c', 'k', 'y']):
    #     intercept, coefs = lasso_regression(X, y, alpha=alpha)
    #
    #     plot_polynomial(X, coefs, intercept)

    # intercept, coef = elasticnet_regression(X, y, alpha=0.01)
    # X = np.sort(X, axis=0)  # for plotting it needs to be sorted
    # plt.plot(X, coef[2] * X ** 2 + coef[1] * X + intercept, color='r')

    #plt.show()

    best_epoch, best_model = early_stopping_sgd(X, y)
    print(best_epoch)
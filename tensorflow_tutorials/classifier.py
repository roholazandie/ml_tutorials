from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_blobs
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def generate_data(visualize=False):
    X_values, y_flat = make_blobs(n_features=2, n_samples=800, centers=3, random_state=500)
    y = OneHotEncoder().fit_transform(y_flat.reshape(-1, 1)).todense()
    y = np.array(y)
    if visualize:
        # Optional line: Sets a default figure size to be a bit larger.
        plt.rcParams['figure.figsize'] = (24, 10)
        plt.scatter(X_values[:, 0], X_values[:, 1], c=y_flat, alpha=0.4, s=150)
        plt.show()
    return X_values, y_flat, y


def learn_parameters(X_train, y_train, X_test, y_test):
    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]

    weights_shape = (n_features, n_classes)

    W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))  # Weights of the model

    X = tf.placeholder(dtype=tf.float32)

    Y_true = tf.placeholder(dtype=tf.float32)

    bias_shape = (1, n_classes)
    b = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(bias_shape))

    Y_pred = tf.matmul(X, W) + b

    loss_function = tf.losses.softmax_cross_entropy(Y_true, Y_pred)
    gdo_learner = tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss_function)
    #adam_learner = tf.train.AdamOptimizer(0.001).minimize(loss_function)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            sess.run(gdo_learner, feed_dict={X: X_train, Y_true: y_train})
            if i % 100:
                result = sess.run(loss_function, feed_dict={X: X_test, Y_true: y_test})
                print(result)

        y_pred = sess.run(Y_pred, {X: X_test})
        W_final, b_final = sess.run([W, b])

    predicted_y_values = np.argmax(y_pred, axis=1)

    return W_final, b_final, predicted_y_values


def visualize_results(X_values):
    h = 1
    x_min, x_max = X_values[:, 0].min() - 2 * h, X_values[:, 0].max() + 2 * h
    y_min, y_max = X_values[:, 1].min() - 2 * h, X_values[:, 1].max() + 2 * h
    x_0, x_1 = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
    decision_points = np.c_[x_0.ravel(), x_1.ravel()]
    # # We recreate our model in NumPy
    Z = np.argmax(np.matmul(decision_points, W_final) + b_final, axis=1)
    #
    # # Create a contour plot of the x_0 and x_1 values
    Z = Z.reshape(x_0.shape)
    plt.contourf(x_0, x_1, Z, alpha=0.5)
    #
    #plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_flat, alpha=0.3)
    #plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted_y_values, marker='x', s=200)
    #
    plt.xlim(x_0.min(), x_0.max())
    plt.ylim(x_1.min(), x_1.max())
    plt.show()
    return


X_values, y_flat, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X_values, y, test_size=0.2)
X_test += np.random.randn(*X_test.shape) * 1.5

W_final, b_final, predicted_y_values = learn_parameters(X_train, y_train, X_test, y_test)
print(predicted_y_values)
visualize_results(X_values)


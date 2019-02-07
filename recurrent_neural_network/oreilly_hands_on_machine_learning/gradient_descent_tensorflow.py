from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
print(np.shape(housing.data))


def manually_computing_gradient_descent(X_data, y_data):
    n, m = np.shape(X_data)
    X_data = np.c_[np.ones((n,1)), X_data]
    #y_data = housing.target

    X = tf.constant(X_data, dtype=tf.float32, name="X")
    y = tf.constant(y_data.reshape(-1, 1), dtype=tf.float32, name="y")
    etha = 0.08#tf.constant(0.08, dtype=tf.float32, name="etha")
    theta = tf.Variable(tf.random_uniform(shape=(m+1, 1), minval=-10, maxval=10), dtype=tf.float32)

    error = tf.matmul(X, theta) -y
    mse_error = tf.reduce_mean(tf.square(error), name="mse")
    gradients = (2/n)* tf.matmul(tf.transpose(X), error)
    training_op = tf.assign(theta, theta - etha * gradients)

    mse_errors = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            sess.run(training_op)
            mserror_val = sess.run(mse_error)
            gradient_val = sess.run(gradients)
            print(mserror_val)
            mse_errors.append(mserror_val)


    plt.plot(mse_errors)
    plt.show()



def autodiff_gradient_descent(X_data, y_data):
    '''
    the same code as above just one line is different:
    tf.gradients(mse_error, [theta])[0]
    we use tf.gradient to compute gradients
    :param X_data:
    :param y_data:
    :return:
    '''
    n, m = np.shape(X_data)
    X_data = np.c_[np.ones((n, 1)), X_data]
    # y_data = housing.target

    X = tf.constant(X_data, dtype=tf.float32, name="X")
    y = tf.constant(y_data.reshape(-1, 1), dtype=tf.float32, name="y")
    etha = 0.08  # tf.constant(0.08, dtype=tf.float32, name="etha")
    theta = tf.Variable(tf.random_uniform(shape=(m + 1, 1), minval=-10, maxval=10), dtype=tf.float32)

    error = tf.matmul(X, theta) - y
    mse_error = tf.reduce_mean(tf.square(error), name="mse")
    gradients = tf.gradients(mse_error, [theta])[0]
    training_op = tf.assign(theta, theta - etha * gradients)

    mse_errors = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            sess.run(training_op)
            mserror_val = sess.run(mse_error)
            gradient_val = sess.run(gradients)
            print(mserror_val)
            mse_errors.append(mserror_val)

    plt.plot(mse_errors)
    plt.show()


def using_optimizer(X_data, y_data):
    n, m = np.shape(X_data)
    X_data = np.c_[np.ones((n, 1)), X_data]
    # y_data = housing.target

    X = tf.constant(X_data, dtype=tf.float32, name="X")
    y = tf.constant(y_data.reshape(-1, 1), dtype=tf.float32, name="y")
    etha = 0.001  # tf.constant(0.08, dtype=tf.float32, name="etha")
    theta = tf.Variable(tf.random_uniform(shape=(m + 1, 1), minval=-10, maxval=10), dtype=tf.float32)

    error = tf.matmul(X, theta) - y
    mse_error = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=etha)
    #optimizer = tf.train.AdamOptimizer(learning_rate=etha)
    training_op = optimizer.minimize(mse_error)

    mse_errors = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100000):
            sess.run(training_op)
            mserror_val = sess.run(mse_error)
            print(mserror_val)
            mse_errors.append(mserror_val)


def mini_batch_gradient_descent(X_data, y_data):
    n, m = np.shape(X_data)
    X_data = np.c_[np.ones((n, 1)), X_data]

    X = tf.placeholder(dtype=tf.float32, name="X")
    y = tf.placeholder(dtype=tf.float32, name="y")
    etha = 0.001  # tf.constant(0.08, dtype=tf.float32, name="etha")
    theta = tf.Variable(tf.random_uniform(shape=(m + 1, 1), minval=-10, maxval=10), dtype=tf.float32)

    error = tf.matmul(X, theta) - y
    mse_error = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=etha)
    training_op = optimizer.minimize(mse_error)

    batch_size = 10
    mse_errors = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100000):

            X_batch = X_data[i:i+batch_size,:].reshape(batch_size, m+1)
            y_batch = y_data[i:i+batch_size,:].reshape(batch_size, 1)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            # mserror_val = sess.run(mse_error, feed_dict={X: , y: })
            # print(mserror_val)
            # mse_errors.append(mserror_val)



if __name__ == "__main__":
    X_data = preprocessing.scale(housing.data)
    standard_scaler = StandardScaler()
    X_data = standard_scaler.fit(housing.data).transform(housing.data)
    manually_computing_gradient_descent(X_data, y_data=housing.target)
    #autodiff_gradient_descent(X_data, y_data=housing.target)
    #using_optimizer(X_data, y_data=housing.target)
    #mini_batch_gradient_descent(X_data, y_data=housing.target)
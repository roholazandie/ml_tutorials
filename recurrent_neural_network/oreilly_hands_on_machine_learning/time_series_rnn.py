import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import fully_connected


def generate_dataset(batch_size, num_steps):
    t = np.linspace(0, 30, 10000)
    y = t * np.sin(t) / 3 + 2 * np.sin(5 * t)
    y = np.reshape(y, [batch_size, len(y) // batch_size])

    for i in range(np.shape(y)[1]//num_steps-1):
        yield np.reshape(y[:, i*(num_steps): (i+1)*num_steps], [batch_size, num_steps, 1]),\
              np.reshape(y[:, i*(num_steps)+1: (i+1)*num_steps+1], [batch_size, num_steps, 1])




def rnn_time_series_approach1():
    '''
    In this approach we try to get an output with OutputProjectionWrapper
    which adds a fully connected layer of linear transformation at the end
    but without activation function
    '''
    # Parameters
    n_steps = 20
    n_inputs = 1
    n_outputs = 1
    n_neurons = 100
    learning_rate = 0.001
    n_iteration = 100
    batch_size = 5

    #Model
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_outputs])

    cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(n_neurons, activation=tf.nn.relu),
                                                  output_size=n_outputs)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    loss = tf.reduce_mean(tf.square(outputs-y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    #Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(n_iteration):
            for X_batch, y_batch in generate_dataset(batch_size=batch_size, num_steps=n_steps):
                loss_value, _ = sess.run([loss, train_op], feed_dict={X: X_batch, y: y_batch})
                print(loss_value)


def rnn_time_series_approach2():
    # Parameters
    n_steps = 20
    n_inputs = 1
    n_outputs = 1
    n_neurons = 100
    learning_rate = 0.001
    n_iteration = 100
    batch_size = 5

    # Model
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_outputs])

    cell = tf.contrib.rnn.BasicRNNCell(n_neurons, activation=tf.nn.relu)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(outputs, [-1, n_neurons])
    stacked_outputs = fully_connected(stacked_rnn_outputs, n_outputs, activation_fn=None)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(n_iteration):
            for X_batch, y_batch in generate_dataset(batch_size=batch_size, num_steps=n_steps):
                loss_value, _ = sess.run([loss, train_op], feed_dict={X: X_batch, y: y_batch})
                print(loss_value)


def creative_rnn():
    # Parameters
    n_steps = 20
    n_inputs = 1
    n_outputs = 1
    n_neurons = 100
    learning_rate = 0.001
    n_iteration = 100
    batch_size = 5
    n_generate = 1000

    # Model
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_outputs])

    cell = tf.contrib.rnn.BasicRNNCell(n_neurons, activation=tf.nn.relu)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(outputs, [-1, n_neurons])
    stacked_outputs = fully_connected(stacked_rnn_outputs, n_outputs, activation_fn=None)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    # Training
    seed_sequence = [0.]*n_steps

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(n_iteration):
            for X_batch, y_batch in generate_dataset(batch_size=batch_size, num_steps=n_steps):
                loss_value, _ = sess.run([loss, train_op], feed_dict={X: X_batch, y: y_batch})

        # Generation
        for iteration in range(n_generate):
            X_batch = np.array(seed_sequence[-n_steps:]).reshape(1, n_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
            seed_sequence.append(y_pred[0, -1, 0])

        #Plot the result
        plt.plot(seed_sequence)
        plt.show()


if __name__ == "__main__":
    #rnn_time_series_approach1()
    #rnn_time_series_approach2()
    creative_rnn()
    t = np.linspace(0, 30, 10000)
    y = t * np.sin(t) / 3 + 2 * np.sin(5 * t)
    plt.plot(t, y)
    plt.show()
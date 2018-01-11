#from tensorflow.python.layers.core import fully_connected
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import fully_connected
import numpy as np
from plotlyvisualization import multi_histogram

def leaky_relu(z, name=None):
    return tf.maximum(0.01*z, z, name=name)


def raw_run():
    '''
    this function assumes no activation function or regularizer
    :return:
    '''
    mnist = input_data.read_data_sets("/tmp/data/")

    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 200
    n_outputs = 10
    learning_rate = 0.05
    n_epochs = 40
    batch_size = 50

    X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(dtype=tf.int64, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
        logits = fully_connected(hidden2, n_outputs, scope="output")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients)
        #train_op = optimizer.minimize(loss)

    with tf.name_scope("evaluation"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    hist1 = []
    hist2 = []
    hist3 = []
    hist4 = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            for iterraion in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(train_op, feed_dict={X: X_batch, y: y_batch})

                gradients_value = sess.run(gradients, feed_dict={X: X_batch, y: y_batch})
                last_bias = gradients_value[-1][0]
                hist1.append(last_bias[0])
                hist2.append(last_bias[1])
                hist3.append(last_bias[2])
                hist4.append(last_bias[3])

            accuracy_val_train = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
            accuracy_val_test = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
            # log_val = sess.run(logits, feed_dict={X: X_batch, y: y_batch})
            # print(log_val)
            #print("train_accuracy", accuracy_val_train, "\ttest_accuracy", accuracy_val_test)




    hists = [hist1, hist2, hist3, hist4]
    multi_histogram(hists)


def activation_function_run():
    mnist = input_data.read_data_sets("/tmp/data/")

    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 200
    n_outputs = 10
    learning_rate = 0.05
    n_epochs = 400
    batch_size = 50

    X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(dtype=tf.int64, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = fully_connected(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = fully_connected(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        logits = fully_connected(hidden2, n_outputs, name="output", activation=tf.nn.elu)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

    with tf.name_scope("evaluation"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            for iterraion in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(train_op, feed_dict={X: X_batch, y: y_batch})

            accuracy_val_train = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
            accuracy_val_test = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
            # log_val = sess.run(logits, feed_dict={X: X_batch, y: y_batch})
            # print(log_val)
            print("train_accuracy", accuracy_val_train, "\ttest_accuracy", accuracy_val_test)


def batch_normalization_run():
    mnist = input_data.read_data_sets("/tmp/data/")

    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 200
    n_outputs = 10
    learning_rate = 0.05
    n_epochs = 400
    batch_size = 50

    X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(dtype=tf.int64, shape=(None), name="y")

    is_training = tf.placeholder(dtype=tf.bool, shape=(), name="is_training")
    batch_norm_param = {
        'is_training': is_training,
        'decay': 0.99,
        'updates_collections': None
    }



    with tf.name_scope("dnn"):
        hidden1 = fully_connected(X, n_hidden1, normalizer_fn=batch_norm, normalizer_params=batch_norm_param, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
        logits = fully_connected(hidden2, n_outputs, scope="output")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

    with tf.name_scope("evaluation"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            for iterraion in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(train_op, feed_dict={X: X_batch, y: y_batch, is_training: True})

            accuracy_val_train = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch, is_training: False})
            accuracy_val_test = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels, is_training: False})
            # log_val = sess.run(logits, feed_dict={X: X_batch, y: y_batch})
            # print(log_val)
            print("train_accuracy", accuracy_val_train, "\ttest_accuracy", accuracy_val_test)



def clipping_gradient_run():
    '''
        this function uses the clipping gradient method to prevent exploding gradients problem
        :return:
        '''
    mnist = input_data.read_data_sets("/tmp/data/")

    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 200
    n_outputs = 10
    learning_rate = 0.05
    n_epochs = 40
    batch_size = 50
    gradient_threshold = 1.0

    X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(dtype=tf.int64, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
        logits = fully_connected(hidden2, n_outputs, scope="output")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        # the gradients is a tensor and clip_by_value actually clip all the elements in that tensor
        clipped_gradients = [(tf.clip_by_value(gradient, -gradient_threshold, gradient_threshold), variable) for (gradient, variable) in gradients]
        train_op = optimizer.apply_gradients(clipped_gradients)

    with tf.name_scope("evaluation"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    hist1 = []
    hist2 = []
    hist3 = []
    hist4 = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            for iterraion in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(train_op, feed_dict={X: X_batch, y: y_batch})

                gradients_value = sess.run(gradients, feed_dict={X: X_batch, y: y_batch})
                last_bias = gradients_value[-1][0]
                hist1.append(last_bias[0])
                hist2.append(last_bias[1])
                hist3.append(last_bias[2])
                hist4.append(last_bias[3])

            accuracy_val_train = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
            accuracy_val_test = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
            # log_val = sess.run(logits, feed_dict={X: X_batch, y: y_batch})
            # print(log_val)
            print("train_accuracy", accuracy_val_train, "\ttest_accuracy", accuracy_val_test)

    hists = [hist1, hist2, hist3, hist4]
    multi_histogram(hists)



if __name__ == "__main__":
    #raw_run()
    #batch_normalization_run()
    # multi_histogram(plot_data=[[3,2,1],[3,4,2],[4,2,1],[4,2,1]])
    clipping_gradient_run()


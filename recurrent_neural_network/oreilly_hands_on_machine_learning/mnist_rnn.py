import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

def image_classification():
    #parameters
    n_steps = 28
    n_inputs = 28
    n_neurons = 150
    n_outputs = 10
    learning_rate = 0.001

    # Data
    mnist = input_data.read_data_sets("../MNIST-data")
    X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
    y_test = mnist.test.labels

    # Model
    X = tf.placeholder(shape=[None, n_steps, n_inputs], dtype=tf.float32)
    y = tf.placeholder(shape=[None], dtype=tf.int32)


    basic_cell = tf.contrib.rnn.BasicRNNCell(n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    logits = fully_connected(states, num_outputs=n_outputs, activation_fn=None)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    loss = tf.reduce_mean(xentropy)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct = tf.nn.in_top_k(predictions=logits, targets=y, k=1)
    accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))


    #training
    n_epoch = 100
    batch_size = 150
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run([train_op], feed_dict={X: X_batch, y: y_batch})
            accuracy_train_value = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
            accuracy_test_value = sess.run(accuracy, feed_dict={X: X_test, y: y_test})
            print("accuracy train: ", accuracy_train_value, "accuracy test: ", accuracy_test_value)



if __name__ == "__main__":
    image_classification()
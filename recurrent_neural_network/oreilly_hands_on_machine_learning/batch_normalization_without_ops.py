import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size,
                                             activation_fn=None,
                                             scope=scope)


def dense_batch_relu(x, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, 100,
                                               activation_fn=None,
                                               scope='dense')

        h2 = batch_norm(h1, updates_collections=None, center=True, scale=True, is_training=phase, scope='bn')

        return tf.nn.relu(h2, 'relu')


tf.reset_default_graph()
x = tf.placeholder('float32', (None, 784), name='x')
y = tf.placeholder('float32', (None, 10), name='y')
phase = tf.placeholder(tf.bool, name='phase')

h1 = dense_batch_relu(x, phase, 'layer1')
h2 = dense_batch_relu(h1, phase, 'layer2')
logits = dense(h2, 10, 'logits')

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)),
        'float32'))

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))


def train():
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    history = []
    iterep = 500
    for i in range(iterep * 30):
        x_train, y_train = mnist.train.next_batch(100)
        sess.run(train_step,
                 feed_dict={x: x_train,
                            y: y_train,
                            phase: 1})
        if (i + 1) % iterep == 0:
            epoch = (i + 1) / iterep
            tr = sess.run([loss, accuracy],
                          feed_dict={x: mnist.train.images,
                                     y: mnist.train.labels,
                                     phase: 1})
            t = sess.run([loss, accuracy],
                         feed_dict={x: mnist.test.images,
                                    y: mnist.test.labels,
                                    phase: 0})
            history += [[epoch] + tr + t]
            print(history[-1])
    return history


if __name__ == "__main__":
    train()
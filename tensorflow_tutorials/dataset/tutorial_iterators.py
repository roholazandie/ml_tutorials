import tensorflow as tf
import numpy as np

# Extracting MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_val, y_val     = mnist.validation.images, mnist.validation.labels
X_test, y_test   = mnist.test.images, mnist.test.labels


epochs = 10
batch_size = 64
iterations = len(y_train) * epochs

# dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#
# dataset = dataset.repeat(epochs).batch(batch_size)
# iterator = dataset.make_one_shot_iterator()
#
# X_data, y_data = iterator.get_next()
# y_data = tf.cast(y_data, tf.int32)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     try:
#         while True:
#             X_data_value = sess.run(X_data)
#             print(np.mean(X_data_value))
#             sess.run(y_data)
#     except tf.errors.OutOfRangeError:
#         pass


'''
In One-shot iterator, we had the shortfall of repetition of same
training dataset in memory and there was absence of periodically 
validating our model using validation dataset in our code. In initializable iterator we overcome these problems.
Initializable iterator has to be initialized with dataset before it starts running.
'''

placeholder_X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
placeholder_y = tf.placeholder(tf.int32, shape=[None])

dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))

dataset = dataset.repeat(epochs).batch(batch_size)
iterator = dataset.make_initializable_iterator()

X_data, y_data = iterator.get_next()
y_data = tf.cast(y_data, tf.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    sess.run(iterator.initializer, feed_dict={placeholder_X: X_train, placeholder_y: y_train})
    try:
        while True:
            X_data_value = sess.run(X_data)
            print(np.mean(X_data_value))
            sess.run(y_data)
    except tf.errors.OutOfRangeError:
        pass


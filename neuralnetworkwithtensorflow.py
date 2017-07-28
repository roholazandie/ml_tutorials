from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
t = tf.placeholder(tf.float32, [10])
y = tf.nn.softmax(2 * tf.matmul(x, W) + b + t)
y_ = tf.placeholder(tf.float32, [None, 10])

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

#Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] paramater
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
print(session.run(cross_entropy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels,
                                            t: np.ones((10,))}))

y_log_y = y_ * tf.log(y) # element-wise product
print(y_log_y)
print(np.shape(session.run(y_log_y, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels,
                                            t: np.ones((10,))})))

# print(session.run(cross_entropy, feed_dict={x: mnist.test.images,
#                                             y_: mnist.test.labels,
#                                             }))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


for _ in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, t: np.ones((10,))})

# Evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(session.run(accuracy, feed_dict={x: mnist.test.images,
                                       y_: mnist.test.labels,
                                       t: np.ones((10,))}))

print(session.run(correct_prediction, feed_dict={x: mnist.test.images,
                                                 y_: mnist.test.labels,
                                                 t: np.ones((10,))}))

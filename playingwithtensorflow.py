import tensorflow as tf
import numpy as np


x = tf.placeholder(tf.float32, [None, 3])
W = tf.Variable(tf.random_normal([3, 4], 100))
b = tf.Variable(tf.zeros([4]))
affine_transformation = tf.matmul(x, W) + b
y = tf.nn.softmax(affine_transformation)
y_ = tf.placeholder(tf.float32, [None, 4])

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

print(session.run(cross_entropy, feed_dict={x: np.ones((1, 3)),
                                            y_: np.ones((1, 4))}))


print(session.run(affine_transformation, feed_dict={x:np.ones((1, 3))}))


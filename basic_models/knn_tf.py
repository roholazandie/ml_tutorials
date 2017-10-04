import numpy as np
import tensorflow as tf


W = [3.0,8.0]
regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
rr = 0.5*tf.pow(tf.norm(W), 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    d = sess.run(regularization_loss)
    d2 = sess.run(rr)
    print(d)
    print(d2)
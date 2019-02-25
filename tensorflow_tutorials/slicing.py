import tensorflow as tf
import numpy as np

ph = tf.placeholder(shape=[None,3], dtype=tf.int32)

# look the -1 in the first position
# the first argument is the matrix(tensor) the second one is the position of
# left-up corner of slice(which is a rectangular) and the last one is the size of ndims
# and shows how many we should proceed in each direction.
x = tf.slice(ph, [0, 0], [1, 2])

input_ = np.array([[1,2,3],
                   [3,4,5],
                   [5,6,7]])

with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(x, feed_dict={ph: input_}))
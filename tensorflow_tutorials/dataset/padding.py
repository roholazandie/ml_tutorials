import tensorflow as tf

t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[2, 3,], [1, 5]])
# 'constant_values' is 0.
# rank of 't' is 2.
result = tf.pad(t, paddings)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    value = sess.run(result)
    print(value)




mat = tf.constant([[ 1,  1,  2, 3],
                 [-1, 2,  1, 2],
                 [-2, -1,  3, 1],
                 [-3, -2, -1, 4]])
out = tf.matrix_band_part(mat, -1, 0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    value = sess.run(out)
    print(value)
import tensorflow as tf

x = tf.constant(1)
y = tf.constant(1)

assertion = tf.assert_equal(x, y)

with tf.control_dependencies([assertion]):
    n = tf.add(x, y)

sess = tf.Session()
print(sess.run(x))
print(sess.run(n))


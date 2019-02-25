import tensorflow as tf
import numpy as np


x = tf.constant(1)
y = tf.constant(1)

assertion = tf.assert_equal(x, y)

with tf.control_dependencies([assertion]):
    n = tf.add(x, y)

sess = tf.Session()
print(sess.run(x))
print(sess.run(n))


'''
In case this wasn't obvious to anyone else, the docs explicitly state that 
"the control dependencies context applies only to ops that are constructed within the context.
 Merely using an op or tensor in the context does not add a control dependency
'''

x = tf.get_variable(name="x", shape=(), dtype=tf.float32)
#z = tf.get_variable(name="z", shape=(), dtype=tf.float32)
op = tf.assign_add(x, 1)

#op2 = tf.assign_add(z, np.random.rand())

with tf.control_dependencies([op]):
    #here we should have op again
    y = tf.identity(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        print(y.eval())

#changing the name
x=tf.identity(x, "cc")
print(x)
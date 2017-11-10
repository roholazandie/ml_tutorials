import tensorflow as tf
import numpy as np

x1 = tf.Variable(initial_value=tf.random_uniform([1], -10, 10))
y1 = tf.Variable(initial_value=tf.random_uniform([1], -10, 10))

session = tf.InteractiveSession()
tf.global_variables_initializer().run()


z = tf.pow(x1, 2) + tf.pow(y1, 2)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(z)
for step in range(1000):
    session.run(train_step)
    print(session.run(z))

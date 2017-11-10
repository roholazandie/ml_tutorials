import tensorflow as tf
import numpy as np

'''
lipschitz constant is a theoretical bound on the step size
for example for f(x)=x^2 the lipschitz constant is C=2
here we can't increase the learning_rate bigger or equal to 1 because it stay in the same place
'''

X= tf.Variable(tf.constant(10.), name='X', dtype=tf.float32)

loss = tf.pow(X, 2)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
num_epochs = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        loss_value, _= sess.run([loss, train_step])
        print(loss_value)

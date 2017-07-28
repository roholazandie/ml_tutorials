import tensorflow as tf

#y = (1 - x1)^2 + 100 * (x2 - x1^2)^2

def clipped_gradients(optimizer, z, x1, x2):
    grads_and_vars = optimizer.compute_gradients(z, [x1, x2])
    clipped_grads_and_vars = [(tf.clip_by_value(g, -10, 10), v) for g, v in grads_and_vars]
    optimizer.apply_gradients(clipped_grads_and_vars)
    optimizer_variable = optimizer.minimize(z)
    return optimizer_variable

def default_gradient(optimizer, z):
    optimizer_variable = optimizer.minimize(z)
    return optimizer_variable

x1 = tf.Variable(initial_value=tf.random_uniform([1], 0, 1))
x2 = tf.Variable(initial_value=tf.random_uniform([1], 0, 1))

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

one_constant = tf.constant(1, dtype=tf.float32)
z = tf.pow(tf.subtract(one_constant, x1), 2) + 100*tf.pow(tf.pow(tf.subtract(x2, x1), 2), 2)

optimizer = tf.train.GradientDescentOptimizer(0.01)
#optimizer_variable = clipped_gradients(optimizer, z, x1, x2)
optimizer_variable = default_gradient(optimizer, z)

'''
grads_and_vars = optimization.compute_gradients(z, [x1, x2])
#print(session.run(grad_and_vars))
#clipped_grads_and_vars = [tf.clip_by_value(g, -10, 10) for g, v in grads_and_vars]
clipped_grads_and_vars = [(tf.clip_by_value(g, -10, 10), v) for g, v in grads_and_vars]
#print(session.run(clipped_grads_and_vars))
optimization.apply_gradients(clipped_grads_and_vars)
train = optimization.minimize(z)
'''

# gradients = tf.gradients(z, [x1, x2])
# print(session.run(gradients))


for step in range(1000):
    session.run(optimizer_variable)
    #print(session.run(z))
    #print(session.run(x1))
    print(session.run(x2))

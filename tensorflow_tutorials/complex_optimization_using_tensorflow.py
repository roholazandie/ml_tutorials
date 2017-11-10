import tensorflow as tf
from plotlyvisualization import animate_optimization, countor
import numpy as np

#y = (1 - x1)^2 + 100 * (x2 - x1^2)^2

session = tf.InteractiveSession()


def apply_clipped_gradients(optimizer, z, x1, x2):
    grads_and_vars = optimizer.compute_gradients(z, [x1, x2])
    clipped_grads_and_vars = [(tf.clip_by_value(g, -10, 10), v) for g, v in grads_and_vars]
    optimizer.apply_gradients(clipped_grads_and_vars)
    optimizer_variable = optimizer.minimize(z)
    return optimizer_variable


def apply_default_gradient(optimizer, z):
    optimizer_variable = optimizer.minimize(z)
    return optimizer_variable


def get_gradients(optimizer, z, x1, x2):
    grads_and_vars = optimizer.compute_gradients(z, [x1, x2])
    clipped_grads_and_vars = [(tf.clip_by_value(g, -10, 10), v) for g, v in grads_and_vars]
    gradients = [g for (g, v) in clipped_grads_and_vars]
    return gradients


def get_variables(optimizer, z, x1, x2):
    grads_and_vars = optimizer.compute_gradients(z, [x1, x2])
    clipped_grads_and_vars = [(tf.clip_by_value(g, -10, 10), v) for g, v in grads_and_vars]
    variables = [v for (g, v) in clipped_grads_and_vars]
    return variables


x1 = tf.Variable(initial_value=tf.random_uniform([1], 0, 1))
x2 = tf.Variable(initial_value=tf.random_uniform([1], 0, 1))

one_constant = tf.constant(2, dtype=tf.float32)

#z = tf.pow(tf.subtract(one_constant, x1), 2) + 100*tf.pow(tf.pow(tf.subtract(x2, x1), 2), 2)
z = tf.pow(x1, 2) + tf.pow(tf.subtract(x2, 1), 2) + tf.sin(3*x1*x2)

print(z.eval(feed_dict={x1: [1], x2: [3]}))  # the value of z in (1,3)


session.run(tf.global_variables_initializer())
optimizer = tf.train.GradientDescentOptimizer(0.01)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9,
#           beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')


gradients = get_gradients(optimizer, z, x1, x2)
variables = get_variables(optimizer, z, x1, x2)

optimizer_variable = apply_clipped_gradients(optimizer, z, x1, x2)


N_iteration = 1000
xs = np.zeros((N_iteration,))
ys = np.zeros((N_iteration,))

gradxs = np.zeros((N_iteration,))
gradys = np.zeros((N_iteration,))

for step in range(1000):
    session.run(optimizer_variable)
    #session.run(optimizer_variable)
    #print(session.run(z))
    (x, y) = session.run(variables)
    xs[step] = x[0]
    ys[step] = y[0]

    (gradx, grady) = session.run(gradients)
    gradxs[step] = gradx[0]
    gradys[step] = grady[0]
    #print(session.run(x1))
    #print(session.run(x2))

xm = np.min(xs) - 1.5
xM = np.max(xs) + 1.5
ym = np.min(ys) - 1.5
yM = np.max(ys) + 1.5

delta = 0.25
x = np.arange(xm, xM, delta)
y = np.arange(ym, yM, delta)
X, Y = np.meshgrid(x, y)
f = lambda x,y: x**2+y**2+np.sin(3*x*y)
Z = f(X,Y)
print(np.shape(Z))
#animate_optimization(x, y, Z, xs, ys, gradxs, gradys)
#countor(x,y,Z)
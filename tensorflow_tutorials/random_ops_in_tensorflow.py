from tensorflow.python.ops.init_ops import RandomUniform
import tensorflow as tf

# RandomUniform is an ops so it represent a node in the graph
# the node doesn't have any shape it is implemented in the __call__ method
# so we can have random_ops without even specifying the shape of them
random_uniform = RandomUniform(minval=-5, maxval=5)
# using __call__ and simply passing argument is the same thing
#random_tensor = random_uniform.__call__(shape=(2, 3), dtype=tf.float32)
float_random_tensor = random_uniform(shape=(1, 3), dtype=tf.float32)
int_random_tensor = random_uniform(shape=(1, 3), dtype=tf.int32)

session = tf.Session()
# I call it realized because when you start a session then you can
# make it visible in python i.e. the value of the tensor come back to python
# from the background
realized_tensor = session.run([int_random_tensor, float_random_tensor])
print(realized_tensor)

# This one is another way to use the RandomUniform class
rand = tf.random_uniform(shape=(1, 3), minval=-5, maxval=5, dtype=tf.float32)
realized_rand = session.run(rand)
print(realized_rand)

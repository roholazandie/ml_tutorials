import tensorflow as tf


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

session = tf.Session()
print(session.run([node1, node2]))

node3 = tf.add(node1, node2)
print(session.run(node3))
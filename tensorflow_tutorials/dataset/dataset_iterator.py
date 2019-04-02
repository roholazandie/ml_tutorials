import tensorflow as tf


dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([10, 2]))
# print(dataset1.output_types)
# print(dataset1.output_shapes)
# print(dataset1.output_classes)

def func(x):
    xx = tf.greater(x, 1.5)
    #print(tf.shape(xx))
    return tf.logical_and(xx[0], xx[1])

dataset1 = dataset1.map(lambda x: x+1)
dataset1 = dataset1.filter(func)

iterator = dataset1.make_initializable_iterator()

X_data = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)

    try:
        while True:
            X_data_value = sess.run(X_data)
            print(X_data_value)
    except tf.errors.OutOfRangeError:
        pass
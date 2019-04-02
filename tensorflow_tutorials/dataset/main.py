import tensorflow as tf
import numpy as np
'''
feed_dict mechanism practice has been advised by Tensorflow developers
to be strongly discontinued either during the training or repeatedly testing same series of dataset.
The only particular scenario in which feed_dict mechanism is to be used is during inferencing of data
during deployment.

this can be replaced by dataset and iterators

'''
tf.enable_eager_execution()
# to emit data as slices support batching
dataset1 = tf.data.Dataset.from_tensor_slices(tf.range(10, 15))
dataset12 = tf.data.Dataset.from_tensor_slices((tf.range(30, 40, 2), np.arange(10, 20, 2)))


# to emit the whole data, no support for batching
dataset2 = tf.data.Dataset.from_tensors(tf.range(10, 15))


'''
This method is useful in cases where you wish to generate the data at runtime and as such no raw data exists 
with you or in scenarios where your training data is extremely huge and it is not possible to store them in your disk. 
I would strongly encourage people to not use this method for the purpose of generating data augmentations.
'''

def generator(seq_type):
    if seq_type == 1:
        for i in range(10):
            yield i
    elif seq_type == 2:
        for i in range(40):
            yield ('Hi', i)


dataset31 = tf.data.Dataset.from_generator(generator, (tf.int32), args=([1]))
dataset32 = tf.data.Dataset.from_generator(generator, (tf.string, tf.int32), args=([2]))

# for item in dataset32:
#     print(item)


'''
data transformations
'''
data = tf.data.Dataset.from_tensors(tf.range(10, 30))
repeated_data = data.repeat(3)
# for i in repeated_data:
#     print(i)

shuffled_data = data.shuffle(buffer_size=5)
for i in shuffled_data:
    print(i)

mapped_data = data.map(lambda x: 3*x)
for i in mapped_data:
    print(i)

# def filter_fn(x):
#     return tf.reshape(tf.not_equal(x % 5, 1), [1,20])
#
# filtered_data = data.filter(filter_fn)
# print(filtered_data)

batched_data = data.batch(batch_size=3)
for i in batched_data:
    print(i)
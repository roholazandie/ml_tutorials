import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_workers", 3, "number of workers")
flags.DEFINE_integer("worker_index", 0, "worker index")


dataset = tf.data.Dataset.range(6)
dataset = dataset.shard(FLAGS.num_workers, FLAGS.worker_index)


iterator = dataset.make_one_shot_iterator()
res = iterator.get_next()

# Suppose you have 3 workers in total
with tf.Session() as sess:
    for i in range(2):
        print(sess.run(res))

'''
0, 3 on worker 0
1, 4 on worker 1
2, 5 on worker 2
'''
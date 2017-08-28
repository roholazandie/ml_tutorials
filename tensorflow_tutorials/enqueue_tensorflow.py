import tensorflow as tf
'''

'''

epoch_size = 10
i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(10):
        print(sess.run(i))
    coord.request_stop()
    coord.join(threads)
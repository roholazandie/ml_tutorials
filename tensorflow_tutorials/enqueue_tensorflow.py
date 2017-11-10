import tensorflow as tf
import recurrent_neural_network.reader as reader
import numpy as np
'''

'''


def queue_tf():
    epoch_size = 10
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for _ in range(10):
            print(sess.run(i))
        coord.request_stop()
        coord.join(threads)



def using_producers():
    num_steps = 10
    batch_size = 5
    data = [i for i in range(1000)]

    X, Y = reader.ptb_producer(data, num_steps=num_steps, batch_size=batch_size)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord=coord)
        first = True
        i = 0
        X_1 = []
        while True:
            i += 1
            X_realized = sess.run(X)
            Y_realized = sess.run(Y)

            print(X_realized)
            print(Y_realized)

            if first:
                X_1 = X_realized
                first = False
                continue

            if np.array_equal(X_realized, X_1):
                print(i)
                break


        coord.request_stop()
        coord.join()




if __name__ == "__main__":
    using_producers()
    #print(np.random.randint(0,10,1))
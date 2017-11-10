from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

log_dir = "/tmp/tensorflow/mnist/logs/ss"

def simple_neural_network(mnist):
    x = tf.placeholder(tf.float32, [None, 784])  # None corresponds to the batch size, can be of any size
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] paramater
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    print(session.run(cross_entropy, feed_dict={x: mnist.test.images,
                                                y_: mnist.test.labels,
                                                }))

    y_log_y = y_ * tf.log(y)  # element-wise product
    print(y_log_y)
    print(np.shape(session.run(y_log_y, feed_dict={x: mnist.test.images,
                                                   y_: mnist.test.labels,
                                                   })))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


    for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # different syntax for the same purpose
        # train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
        session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(session.run(accuracy, feed_dict={x: mnist.test.images,
                                           y_: mnist.test.labels
                                           }))

    print(session.run(correct_prediction, feed_dict={x: mnist.test.images,
                                                     y_: mnist.test.labels
                                                     }))


def simple_neural_network_with_tensorboard_output(mnist):
    with tf.Graph().as_default():
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, [None, 784], name="input_image")  # None corresponds to the batch size, can be of any size

        with tf.name_scope("layer1"):
            W = tf.Variable(tf.zeros([784, 10]), name="weights")
            b = tf.Variable(tf.zeros([10]), name="bias")
            y = tf.nn.softmax(tf.matmul(x, W) + b)

        with tf.name_scope("output"):
            y_ = tf.placeholder(tf.float32, [None, 10])

        #global_step = tf.Variable(0, name='global_step', trainable=False)

        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)
        # Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] paramater
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name="cross_entropy")

        tf.summary.scalar('loss', cross_entropy)


        train_step = tf.train.GradientDescentOptimizer(0.5, name="GDO").minimize(cross_entropy)
        # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(log_dir, session.graph)

        max_steps = 10000
        for step in range(max_steps):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # different syntax for the same purpose
            # train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
            feed_dict = {x: batch_xs, y_: batch_ys}
            session.run(train_step, feed_dict=feed_dict)
            if step % 100:
                summary_str = session.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(session, checkpoint_file, global_step=step)

        # Evaluation
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(session.run(accuracy, feed_dict={x: mnist.test.images,
                                               y_: mnist.test.labels
                                               }))


if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    simple_neural_network_with_tensorboard_output(mnist)
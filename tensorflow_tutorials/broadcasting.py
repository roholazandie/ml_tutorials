import tensorflow as tf


def broadcast_across_rows():
    a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
    b = tf.constant([100,100,100], name='b')
    add_op = a + b

    with tf.Session() as session:
        print(session.run(add_op))


def broadcast_across_columns():
    a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
    b = tf.constant([[100], [101]], name='b')
    add_op = a + b

    with tf.Session() as session:
        print(session.run(add_op))


if __name__ == "__main__":
    broadcast_across_rows()
    #broadcast_across_columns()
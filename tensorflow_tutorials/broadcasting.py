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


def broadcast():
    a = tf.constant([[[0, 7, 6 ], [7 ,1, 3]],[[8,  3, 1], [9, 9, 7 ]]])

    b = tf.constant([[[2, 8, 6]],[[7 ,1, 7]]])

    c = a+b
    with tf.Session() as sess:

        print(sess.run(a))
        print("------------")
        print(sess.run(b))
        print("------------")
        print(sess.run(c))

def broadcast2():
    tf.enable_eager_execution()
    a = tf.constant([[[0, 7, 6],
                      [7 ,1, 3],
                      [7 ,1, 3]],

                     [[8,  3, 1],
                      [9, 9, 7],
                      [7 ,1, 3]],


                     ], dtype=tf.float32)
    N = tf.shape(a)[0]
    F = tf.shape(a)[1]
    bias = tf.concat([tf.ones(shape=(F, 2), dtype=tf.float32),
                      tf.zeros(shape=(F, 1), dtype=tf.float32)]
                     ,1)
    bias = tf.expand_dims(bias, axis=0)
    result = a + bias
    print(result)

    # more compact broadcasting bias = [1 ,1 , 0]
    bias = tf.concat([tf.ones(shape=(2,), dtype=tf.float32), tf.zeros(shape=(1,), dtype=tf.float32)], axis=0)
    bias = tf.expand_dims(tf.expand_dims(bias, axis=0), axis=0)

    result = a + bias
    print(result)





if __name__ == "__main__":
    #broadcast_across_rows()
    #broadcast_across_columns()
    broadcast2()
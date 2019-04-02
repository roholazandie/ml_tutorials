import tensorflow as tf


def simple_loop_example():
    def cond(t1, t2):
        return tf.less(t1, t2)

    def body(t1, t2):#body should always take and return updated variables of loop_vars
        return [tf.add(t1, 1), t2]

    t1 = tf.constant(1)
    t2 = tf.constant(5)

    res = tf.while_loop(cond, body, loop_vars=[t1, t2])#loop_vars change everytime body is running

    with tf.Session() as sess:
        print(sess.run(res))




def shape_invariance_example():
    '''
    with shape invariance we determine the shapes of input to the body in each iteration
    we have to use it otherwise it doesnt work for the next loops
    :return:
    '''
    i0 = tf.constant(0)
    m0 = tf.ones([2, 2])
    c = lambda i, m: i < 10
    b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
    res = tf.while_loop(
        c, b, loop_vars=[i0, m0],
        shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])

    with tf.Session() as sess:
        result = sess.run(res)
        print(result)


if __name__ == "__main__":
    shape_invariance_example()
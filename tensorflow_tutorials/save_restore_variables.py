import tensorflow as tf
import numpy as np

def save_variables():
    v1 = tf.Variable(tf.random_normal((2,3)), name='v1')
    v2 = tf.Variable(tf.random_normal((2,3)), name='v2')

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver({'var1': v1, 'var2': v2})

    with tf.Session() as session:
        session.run(init_op)
        save_path = saver.save(session, "/tmp/model.ckpt")
        print(save_path)
        print(session.run(v1))


def restore_variables():
    v1 = tf.Variable(tf.zeros((2,3)), name='var1')
    #v2 = tf.Variable(tf.zeros((2,3)), name='var2')

    saver = tf.train.Saver()

    with tf.Session() as session:
        saver.restore(session, "/tmp/model.ckpt")
        print("Model restored")
        print(session.run(v1))

if __name__ == "__main__":
    '''
    save_variables save the variables to a directory and then
    restore variables restore them to the variables
    '''
    #save_variables()
    restore_variables()
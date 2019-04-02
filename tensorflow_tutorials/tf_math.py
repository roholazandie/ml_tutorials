import tensorflow as tf


def sample_top_k_vector():
    data = tf.constant([[1.0, 2.0, 3.0, 1.5]])
    values, indices = tf.math.top_k(data, k=1)

    with tf.Session() as sess:
        result = sess.run(values)
        print(result)

def sample_top_k_matrix():
    data= tf.constant([[3.0, 5.0, 4.0, 7.0, 11.0], [1.0, 9.0, 0.0, 8.0, 10.0]])
    values, indices = tf.math.top_k(data, k=2)# compute the max value of rows

    with tf.Session() as sess:
        result = sess.run(values)
        print(result)


def tfwhere():
    a = tf.constant([[2.0, 1.0], [0.0, 2.0]])
    b = tf.constant([[1.0, 0.0], [3.0, 5.0]])
    c = tf.constant([[1.0, 1.0], [1.0, 1.0]])
    d = tf.constant([[0.0, 0.0], [0.0, 0.0]])
    res = tf.where(a < b, c, d)

    with tf.Session() as sess:
        result = sess.run(res)
        print(result)


def multinomial_sampling():
    logits = tf.constant([[ 0.9, 0.1]])
    res = tf.multinomial(logits, num_samples=100)

    with tf.Session() as sess:
        result = sess.run(res)
        print(result)

if __name__ == "__main__":
    #sample_top_k_vector()
    #sample_top_k_matrix()
    #tfwhere()
    multinomial_sampling()
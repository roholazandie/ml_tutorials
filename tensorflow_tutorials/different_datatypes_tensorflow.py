import tensorflow as tf

log_dir = "/home/rohola/tmp/tutorial_log_dir"

def different_data_types():
    a = tf.zeros(name='a', shape=(2,), dtype=tf.float32)
    b = tf.constant(value=3, name='b', shape=(2,), dtype=tf.float32)
    c = tf.ones(name='c', shape=(2,), dtype=tf.float32)
    d = tf.linspace(name='d', start=1.0, stop=10.0, num=20)
    z = a+b+c
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(z)
        session.run(d)

        summary_writer = tf.summary.FileWriter(log_dir, session.graph)
        summary_writer.flush()

if __name__ == "__main__":
    different_data_types()
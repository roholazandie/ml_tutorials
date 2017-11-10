import tensorflow as tf


def fauilure_of_calling_one_variable_multiple_times():
    def some_variable(shape):
        var = tf.get_variable("var", shape=shape, initializer=tf.random_normal_initializer)
        return var
    # Since the desired behavior is unclear
    # (create new variables or reuse the existing ones?) TensorFlow will fail.
    some_variable((2,3))
    some_variable((2,3))


def sharing_variables():
    '''
    Variable scopes allow you to control variable reuse when calling functions which implicitly create and use variables.
    They also allow you to name your variables in a hierarchical and understandable way.
    '''
    def my_variables():
        with tf.variable_scope("v1"):
            weights = tf.get_variable("weight", shape=(2, 3), initializer=tf.random_uniform_initializer())
        return weights

    def two_scopes_with_same_call_to_variable_creator_fun():
        with tf.variable_scope("scope1") as scope1:
            result1 = my_variables()
            print(result1.name) # scope1/v1/weight:0

        with tf.variable_scope("scope2") as scope2:
            result2 = my_variables()
            print(result2.name) # scope2/v1/weight:0

    def reusing_variable():
        with tf.variable_scope("scope") as scope:
            result1 = my_variables()
            rscope.reuse_variables()  # allow scope1 to use the same variable again
            result2 = my_variables()
            assert result1 is result2
            print(result1.name) # scope1/v1/weight:0
            print(result2.name) # scope1/v1/weight:0

    #two_scopes_with_same_call_to_variable_creator_fun()
    reusing_variable()


def naming_variables():
    with tf.variable_scope("foo"):
        with tf.variable_scope("bar"):
            v = tf.get_variable("v", [1])
    assert v.name == "foo/bar/v:0"


if __name__ == "__main__":
    #fauilure_of_calling_one_variable_multiple_times()
    sharing_variables()

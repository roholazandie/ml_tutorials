from tensorflow.contrib.data.python.ops.dataset_ops import Dataset
import tensorflow as tf
import os
import csv
import pandas as pd
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.svm import SVM


def dataset_reading(filenames):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + "/iris.data.txt"
    dataset = pd.read_csv(filename)
    return dataset


if __name__ == "__main__":
    classifier = 'linearclassifier'
    dataset = dataset_reading('iris.data.txt')
    dataset = dataset.sample(frac=1)
    features_train = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']][0:100]
    labels_train = dataset[['class']][0:100]

    features_test = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']][100:-1]
    labels_test = dataset[['class']][100:-1]

    input_fn_train = tf.estimator.inputs.pandas_input_fn(features_train, labels_train, shuffle=False, num_epochs=10)
    input_fn_test = tf.estimator.inputs.pandas_input_fn(features_test, labels_test, shuffle=False, num_epochs=10)

    sepal_length = tf.feature_column.numeric_column('sepal_length')
    sepal_width = tf.feature_column.numeric_column('sepal_width')
    petal_length = tf.feature_column.numeric_column('petal_length')
    petal_width = tf.feature_column.numeric_column('petal_width')
    # class_label = tf.feature_column.categorical_column_with_vocabulary_list("class",['Iris-setosa',
    #                                                                                  'Iris-versicolor',
    #                                                                                  'Iris-virginica'])

    if classifier == 'linearclassifier':
        estimator = tf.estimator.LinearClassifier(
            feature_columns=[sepal_length, sepal_width, petal_length, petal_width],
            n_classes=3,
            label_vocabulary=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        )
    elif classifier == 'DNNClassifier':
        estimator = tf.estimator.DNNClassifier(feature_columns=[sepal_length, sepal_width, petal_length, petal_width],
                                               n_classes=3,
                                               label_vocabulary=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                                               hidden_units=[10, 10])
    else:
        raise Exception("No valid classifier specified")

    estimator.train(input_fn=input_fn_train, max_steps=100)

    results = estimator.evaluate(input_fn=input_fn_test)

    # Print the stats for the evaluation.
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
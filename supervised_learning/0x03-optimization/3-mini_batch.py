#!/usr/bin/env python3
"""File that contains the function train_mini_batch"""
import numpy as np
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Function that trains a loaded neural network model using mini-batch
    gradient descent:
    Args:
    X_train is a numpy.ndarray of shape (m, 784) containing the training data
        m is the number of data points
        784 is the number of input features
    Y_train is a one-hot numpy.ndarray of shape (m, 10) containing the
    training labels
        10 is the number of classes the model should classify
    X_valid is a numpy.ndarray of shape (m, 784) containing the
    validation data
    Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing
    the validation labels
    batch_size is the number of data points in a batch
    epochs is the number of times the training should pass through the
    whole dataset
    load_path is the path from which to load the model
    save_path is the path to where the model should be saved after training
    Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        m = X_train.shape[0]

        save_data = tf.train.import_meta_graph(load_path + ".meta")
        save_data.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        if (m % batch_size) == 0:
            num_minibatches = int(m / batch_size)
            check = 1
        else:
            num_minibatches = int(m / batch_size) + 1
            check = 0

        for epoch in range(epochs + 1):
            train = {x: X_train, y: Y_train}
            valid = {x: X_valid, y: Y_valid}

            loss_train = sess.run(loss, feed_dict=train)
            accuracy_train = sess.run(accuracy, feed_dict=train)
            loss_valid = sess.run(loss, feed_dict=valid)
            accuracy_valid = sess.run(accuracy, feed_dict=valid)

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(loss_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(loss_valid))
            print("\tValidation Accuracy: {}".format(accuracy_valid))

            if epoch < epochs:
                Xs, Ys = shuffle_data(X_train, Y_train)

                for step_number in range(num_minibatches):
                    start = step_number * batch_size
                    end = (step_number + 1) * batch_size
                    if check == 0 and step_number == num_minibatches - 1:
                        x_minbatch = Xs[start::]
                        y_minbatch = Ys[start::]
                    else:
                        x_minbatch = Xs[start:end]
                        y_minbatch = Ys[start:end]

                    feed_mini = {x: x_minbatch, y: y_minbatch}
                    sess.run(train_op, feed_dict=feed_mini)

                    if ((step_number + 1) % 100 == 0) and (step_number != 0):
                        step_cost = sess.run(loss, feed_dict=feed_mini)
                        step_accuracy = sess.run(accuracy, feed_dict=feed_mini)
                        print("\tStep {}:".format(step_number + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))
        save_path = save_data.save(sess, save_path)

    return save_path

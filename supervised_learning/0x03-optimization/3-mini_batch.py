#!/usr/bin/env python3
"""File that contains the function train_mini_batch"""
import numpy as np
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    Function that trains a loaded neural network model using mini-batch gradient descent:
    Args:
    X_train is a numpy.ndarray of shape (m, 784) containing the training data
        m is the number of data points
        784 is the number of input features
    Y_train is a one-hot numpy.ndarray of shape (m, 10) containing the training labels
        10 is the number of classes the model should classify
    X_valid is a numpy.ndarray of shape (m, 784) containing the validation data
    Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing the validation labels
    batch_size is the number of data points in a batch
    epochs is the number of times the training should pass through the whole dataset
    load_path is the path from which to load the model
    save_path is the path to where the model should be saved after training
    Returns: the path where the model was saved
    """
    m = X_train.shape[0]

    load_data = tf.train.import_meta_graph("{}.meta".format(load_path))
    save_data = tf.train.Saver()

    with tf.Session() as sess:
        load_data.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        for i in range(epochs + 1):
            accuaracy_train = sess.run(
                accuracy, feed_dict={x: X_train, y: Y_train})
            loss_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_valid = sess.run(
                accuracy, feed_dict={x: X_valid, y: Y_valid})
            loss_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(loss_train))
            print("\tTraining Accuracy: {}".format(accuaracy_train))
            print("\tValidation Cost: {}".format(loss_valid))
            print("\tValidation Accuracy: {}".format(accuracy_valid))

            if i < epochs:
                X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)

                counter = 0
                j = 0
                z = batch_size

                while (z <= m):
                    sess.run(train_op, feed_dict={x: X_shuffle[j: z],
                                                  y: Y_shuffle[j: z]})
                if (counter + 1) % 100 == 0 and counter != 0:
                    step_accu = sess.run(accuracy,
                                         feed_dict={x: X_shuffle[j: z],
                                                    y: Y_shuffle[j: z]})
                    step_cost = sess.run(loss, feed_dict={x: X_shuffle[j: z],
                                                          y: Y_shuffle[j: z]})

                    print("\tStep {}:".format(counter + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accu))

                j = z
                if (z + batch_size <= m):
                    z += batch_size
                else:
                    z += m % batch_size

                counter += 1

        return save_data.save(sess, save_path)

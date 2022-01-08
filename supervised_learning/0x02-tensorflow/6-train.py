#!/usr/bin/env python3
""" Train """
import tensorflow.compat.v1 as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha, iterations,
          save_path="/tmp/model.ckpt"):
    """ builds, trains, and saves a neural network classifier.
        X_train: (np.ndarray) containing the training input data.
        Y_train: (np.ndarray) containing the training labels
        X_valid: (np.ndarray) containing the validation input data.
        Y_valid: (np.ndarray) containing the validation labels.
        layer_sizes: (list) with nodes in every layer.
        activations: (list) with activations functions in every layer.
        alpha: (float) with the learning rate.
        iterations: (int) number of iterations to train over.
        save_path: (str) path to save the model.
        Returns: the path where the model was saved.
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    saver = tf.train.Saver()
    initg = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initg)
        for i in range(iterations + 1):
            accu_train = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            loss_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accu_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            loss_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(loss_train))
                print("\tTraining Accuracy: {}".format(accu_train))
                print("\tValidation Cost: {}".format(loss_valid))
                print("\tValidation Accuracy: {}".format(accu_valid))
            if i != iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        return saver.save(sess, save_path)
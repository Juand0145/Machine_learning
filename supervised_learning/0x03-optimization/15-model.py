#!/usr/bin/env python3
"""Script to train a model with a multiple-object optimization theory"""
import numpy as np
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Forward propagation method using TF
    Args:
    x: Input data (placeholder)
    layer_sizes: type list are the n nodes inside the layers
    activations: type list with the activation function per layer
    Returns: Prediction of a DNN
    """
    layer = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        if i != len(layer_sizes) - 1:
            layer = create_batch_norm_layer(layer,
                                            layer_sizes[i],
                                            activations[i])
        else:
            layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer


def shuffle_data(X, Y):
    """
    Function that shuffles the data points in two matrices the same way
    Args:
    X is the first numpy.ndarray of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    Y is the second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y
    Returns: the shuffled X and Y matrices
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    X_shuffled = X[shuffle]
    Y_shuffled = Y[shuffle]
    return X_shuffled, Y_shuffled


def calculate_loss(y, y_pred):
    """
    Method to calculate the cross-entropy loss
    of a prediction
    Args:
    y: input data type label in a placeholder
    y_pred: type tensor that contains the DNN prediction
    Returns:
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def calculate_accuracy(y, y_pred):
    """
    method to calculate the accuracy of a prediction in a DNN
    Args:
    y: input data type label in a placeholder
    y_pred: type tensor that contains the DNN prediction
    Returns: Prediction accuracy
    """
    correct_prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def create_layer(prev, n, activation):
    """
    method to create a TF layer
    Args:
    prev: tensor of the previous layer
    n: n nodes created
    activation: activation function
    Returns: Layer created with shape n
    """
    # Average number of inputs and output connections.
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=initializer,
                                  name='layer')
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """
    Function that  creates a batch normalization layer for a neural
    network in tensorflow
    Args:
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used on the output
    of the layer
    Returns: a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    x = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    x_prev = x(prev)
    scale = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    mean, variance = tf.nn.moments(x_prev, axes=[0])
    offset = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    variance_epsilon = 1e-8

    normalization = tf.nn.batch_normalization(
        x_prev,
        mean,
        variance,
        offset,
        scale,
        variance_epsilon,
    )
    if activation is None:
        return normalization
    return activation(normalization)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Function that creates the training operation for a neural
    network in tensorflow using the Adam optimization algorithm
    Args:
    loss is the loss of the network
    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero
    Returns: the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(alpha,
                                       beta1,
                                       beta2,
                                       epsilon).minimize(loss)
    return optimizer


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that creates a learning rate decay operation in tensorflow
    using inverse time decay
    Args:
    alpha is the original learning rate
    decay_rate is the weight used to determine the rate at which alpha
    will decay
    global_step is the number of passes of gradient descent that have elapsed
    decay_step is the number of passes of gradient descent that should
    occur before alpha is decayed further
    Returns: the learning rate decay operation
    """
    LRD = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                      decay_rate, staircase=True)
    return LRD


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Function that  builds, trains, and saves a neural network model in
    tensorflow using Adam optimization, mini-batch gradient descent, learning
    rate decay, and batch normalization
    Args:
    Data_train is a tuple containing the training inputs and training labels,
    respectively
    Data_valid is a tuple containing the validation inputs and validation
    labels, respectively
    layers is a list containing the number of nodes in each layer of the
    network activation is a list containing the activation
    unctions used for each
    layer of the network
    alpha is the learning rate
    beta1 is the weight for the first moment of Adam Optimization
    beta2 is the weight for the second moment of Adam Optimization
    epsilon is a small number used to avoid division by zero
    decay_rate is the decay rate for inverse time decay of the learning rate
    (the corresponding decay step should be 1)
    batch_size is the number of data points that should be in a mini-batch
    epochs is the number of times the training should pass through the whole
    dataset
    save_path is the path where the model should be saved to
    Returns: the path where the model was saved
    """
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0)
    alpha_op = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_op, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        if (m % batch_size) == 0:
            num_minibatches = int(m / batch_size)
            check = 1
        else:
            num_minibatches = int(m / batch_size) + 1
            check = 0

        for epoch in range(epochs + 1):
            feed_train = {x: X_train, y: Y_train}
            feed_valid = {x: X_valid, y: Y_valid}
            train_cost = sess.run(loss, feed_dict=feed_train)
            train_accuracy = sess.run(accuracy, feed_dict=feed_train)
            valid_cost = sess.run(loss, feed_dict=feed_valid)
            valid_accuracy = sess.run(accuracy, feed_dict=feed_valid)

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

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
            sess.run(tf.assign(global_step, global_step + 1))
            save_path = saver.save(sess, save_path)
    return save_path

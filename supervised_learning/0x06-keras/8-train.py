#!/usr/bin/env python3
"""File that contains the function train_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """
    Function That trains a model using mini-batch gradient descent
    Args:
    network is the model to train
    data is a numpy.ndarray of shape (m, nx) containing the input data
    labels is a one-hot numpy.ndarray of shape (m, classes) containing
    the labels of data
    batch_size is the size of the batch used for mini-batch gradient descent
    epochs is the number of passes through data for mini-batch gradient descent
    validation_data is the data to validate the model with, if not None
    """
    def learning_rate_decay(epoch):
        """Function tha uses the learning rate"""
        alpha_0 = alpha / (1 + (decay_rate * epoch))
        return alpha_0

    callbacks = []

    if validation_data:
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(patience=patience)
            callbacks.append(early_stop)

        if learning_rate_decay:
            decay = K.callbacks.LearningRateScheduler(learning_rate_decay,
                                                      verbose=1)
            callbacks.append(decay)

    if save_best:
        save = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        callbacks.append(save)

    train = network.fit(x=data, y=labels, batch_size=batch_size,
                        epochs=epochs, validation_data=validation_data,
                        callbacks=callbacks,
                        verbose=verbose, shuffle=shuffle)

    return train

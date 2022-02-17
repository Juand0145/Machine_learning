#!/usr/bin/env python3
"""File that contains the function preprocess_data and the model"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    FUnction that pre-processes the data for your model
    Args:
    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the
    CIFAR 10 data, where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.inception_v3.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def model():
    """
    Function that works to clasiffy CIFAR 10
    Args:
    Returns the Model
    """
    inception = K.applications.InceptionV3(include_top=False,
                                           input_shape=(128, 128, 3))
    inception.layers.pop()
    model = K.Sequential()
    model.add(K.layers.UpSampling2D(size=(4, 4)))
    model.add(inception)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(units=128,
                             activation='relu',
                             kernel_initializer='he_normal'))

    model.add(K.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer='he_normal'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def learning_rate_decay(epoch):
    """
    Funtion thats used to generate a learning rate decay
    """
    alpha_utd = 0.001 / (1 + (1 * epoch))
    return alpha_utd


if __name__ == '__main__':
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)

    model = model()

    callbacks = []
    checkpoint = K.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                             save_best_only=True,
                                             monitor='val_loss',
                                             mode='min')
    callbacks.append(checkpoint)

    decay = K.callbacks.LearningRateScheduler(learning_rate_decay,
                                              verbose=1)
    callbacks.append(decay)

    EarlyStopping = K.callbacks.EarlyStopping(patience=3,
                                              monitor='val_loss',
                                              mode='min')
    callbacks.append(EarlyStopping)

    model.fit(X_train,
              Y_train,
              batch_size=100,
              epochs=2,
              verbose=True,
              shuffle=False,
              validation_data=(X_valid, Y_valid),
              callbacks=callbacks)
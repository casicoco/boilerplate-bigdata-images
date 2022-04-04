import tensorflow as tf
from keras.layers import Dense, SimpleRNN, Reshape
from ts_boilerplate.params import DATA, TRAIN

# TODO: Should we add here the preprocessing? into a class called "pipeline"?
# TODO: Should we refacto in a class ? Probably!


def get_model(X_train, y_train):
    """Instanciate and return the model of your choice"""
    # $CHALLENGIFY_BEGIN
    model = tf.keras.Sequential()
    model.add(SimpleRNN(1, activation='tanh', input_shape=X_train.shape[1:]))
    model.add(Dense(TRAIN['output_length'] * DATA["n_targets"], activation='linear'))
    model.add(Reshape(y_train.shape[1:]))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), metrics=tf.keras.metrics.MAPE)
    return model
    # $CHALLENGIFY_END


def fit_model(model, X_train, y_train, **kwargs):
    """Fit the `model` object, including preprocessing if needs be"""
    # $CHALLENGIFY_BEGIN
    verbose = kwargs.get("verbose", 0)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=2,
                                          verbose=verbose,
                                          mode='min',
                                          restore_best_weights=True)
    history = model.fit(X_train,
                        y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.3,
                        callbacks=[es],
                        verbose=verbose)
    return history
    # $CHALLENGIFY_END


def predict_output(model, X_test):
    """Return y_test. Include preprocessing if needs be"""
    # $CHALLENGIFY_BEGIN
    y_pred = model.predict(X_test)
    return y_pred
    # $CHALLENGIFY_END

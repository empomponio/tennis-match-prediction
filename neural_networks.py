import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from keras import models, layers
import numpy as np

def get_nn(n_features):
    tf.random.set_seed(123)

    def create_model():
        neurons = 128
        dropout_rate = 0.7
        model = models.Sequential()
        model.add(Dense(units=neurons, activation='relu', input_shape=(n_features,)))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    model = KerasClassifier(model=create_model, verbose=0, epochs=100, optimizer__learning_rate=10.0, batch_size=128)
    return model


def nn_preprocessing(df, cat_col_names):
    cat_cols = pd.get_dummies(data=df[cat_col_names], columns=cat_col_names)
    df_ = df.copy()
    df_.drop(cat_col_names, axis=1)
    df_=(df_-df_.mean())/df_.std()
    df_ = pd.concat([df_, cat_cols], axis=1)
    return df_


def get_nn_grid(n_features):

    neurons = np.logspace(3, 10, num=8, base=2, dtype=int)
    dropout_rates = np.linspace(0.1, 0.8, num=8)
    learning_rates = np.logspace(-4, 1, 6)
    batch_size = np.logspace(4, 12, num=9, base=2, dtype=int)
    epochs = [5, 10, 50, 100]
    
    param_grid = dict(batch_size=batch_size, epochs=epochs, model__dropout_rate=dropout_rates, optimizer__learning_rate=learning_rates, model__neurons=neurons)
    #print(param_grid)

    def create_model(dropout_rate, neurons):
        model = models.Sequential()
        model.add(Dense(units=neurons, activation='relu', input_shape=(n_features,)))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    model = KerasClassifier(model=create_model, verbose=0)
    return (model, param_grid)
import pandas as pd 
import numpy as np
import tensorflow as tf



def create_mlp(dim):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(8, input_dim=dim, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    model.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])
    return model

# %%
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class SensoryRegressor:
    
    def __init__(self, layers_n, input_shape, eta=0.00001, name="regressor"):

        self.name = name
        layers_n = layers_n
        n = len(layers_n)

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(layers_n[0], input_shape=(input_shape, ), activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        for x in range(1, n-1):
            self.model.add(keras.layers.Dense(layers_n[x], activation='relu'))
        self.model.add(keras.layers.Dense(layers_n[n-1], activation='linear'))
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(), 
            optimizer=tf.keras.optimizers.Adam(learning_rate=eta),
            metrics=["accuracy"])
        print(self.model.summary())

    def fit(self, inputs, targets, epochs=2000, batch_size=500):

        History = self.model.fit(inputs,
            targets,
            epochs=epochs,
            batch_size=batch_size)
        np.save(f"{self.name}_fit", [History])

        return History

    def predict(self, inputs):

        return self.model.predict(inputs)

    def save(self, destfile):

        self.model.save(destfile)

    def load(self, srcfile):

        self.model = tf.keras.models.load_model(srcfile)

class SensoryClassificator:
    
    def __init__(self, layers_n, input_shape, eta=0.00001, name= "classifier"):

        self.name = name
        layers_n = layers_n
        n = len(layers_n)

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(layers_n[0], input_shape=(input_shape, ), activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        for x in range(1, n-1):
            self.model.add(keras.layers.Dense(layers_n[x], activation='relu'))
        self.model.add(keras.layers.Dense(layers_n[n-1], activation='softmax'))
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(), 
            optimizer=tf.keras.optimizers.Adam(learning_rate=eta),
            metrics=["accuracy"])
        print(self.model.summary())

    def fit(self, inputs, targets, epochs=2000, batch_size=500):

        History = self.model.fit(inputs,
            targets,
            epochs=epochs,
            batch_size=batch_size)
        np.save(f"{self.name}_fit", [History])

        return History

    def predict(self, inputs):

        return self.model.predict(inputs)

    def save(self, destfile):

        self.model.save(destfile)

    def load(self, srcfile):

        self.model = tf.keras.models.load_model(srcfile)



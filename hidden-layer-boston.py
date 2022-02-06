import tensorflow as tf
import pandas as pd

filepath = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(filepath)

independent = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax','ptratio', 'b', 'lstat']]
dependent = boston[['medv']]
print(independent.shape, dependent.shape)

X = tf.keras.layers.Input(shape=[13])
H = tf.keras.layers.Dense(10, activation='swish')(X)
Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

model.summary()

model.fit(independent, dependent, epochs=100)

print(model.predict(independent[:5]))
print(dependent[:5])

print(model.get_weights())
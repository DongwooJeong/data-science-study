import tensorflow as tf
import pandas as pd

filepath = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(filepath)

iris = pd.get_dummies(iris)

independent = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dependent = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(independent.shape, dependent.shape)

X = tf.keras.layers.Input(shape=[4])
H = tf.keras.layers.Dense(8, activation="swish")(X)
H = tf.keras.layers.Dense(8, activation="swish")(H)
H = tf.keras.layers.Dense(8, activation="swish")(H)
Y = tf.keras.layers.Dense(3, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy',
              metrics='accuracy')

model.summary()

model.fit(independent, dependent, epochs=100)

print(model.predict(independent[:5]))
print(dependent[:5])
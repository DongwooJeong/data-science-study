import tensorflow as tf
import pandas as pd
 
filepath = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(filepath)
lemonade.head()

independent = lemonade[['온도']]
dependent = lemonade[['판매량']]
print(independent.shape, dependent.shape)
 

X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')
 
 
model.fit(independent, dependent, epochs=1000, verbose=0)
model.fit(independent, dependent, epochs=10)
 
 
print(model.predict(independent))
print(model.predict([[15]]))
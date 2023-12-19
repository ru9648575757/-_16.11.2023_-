import tensorflow as tf
from tensorflow import keras
import numpy as np

model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

#y=5x+1
#xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
#ys = np.array([-4.0, 1.0, 6.0, 11.0, 16.0, 21.0], dtype=float)

#xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=int)
#ys = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59], dtype=int)

#y=x**2
xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
ys = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100], dtype=int)

model.fit(xs, ys, epochs=1000)

print(model.predict([11.0]))
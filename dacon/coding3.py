import tensorflow as tf
import numpy as np
from tensorflow import keras

xs = np.array([[1, 2, 3, 4, 5, 6]])
ys = np.array([100, 150, 200, 250, 300, 350])
print(xs.shape)
print(ys.shape)
ys = np.transpose(ys)
print(ys.shape)

# def house_model(y_new):
#     xs = np.array([1, 2, 3, 4, 5, 6])
#     ys = np.array([100, 150, 200, 250, 300, 350])
#     ys = np.transpose(ys)
#     model = keras.models.Sequential()
#     keras.layers.Dense(25, input_dim=6)
#     keras.layers.Dense(50)
#     keras.layers.Dense(25)
#     keras.layers.Dense(1)
#     model.compile(optimizer='adam', loss='mse', metrics=['acc'])
#     model.fit(xs, ys, epochs=30)
#     return model.predict(y_new)[0]

# prediction = house_model([7.0])
# print(prediction)


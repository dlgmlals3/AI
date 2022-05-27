
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 784) / 255.
x_test = x_test.reshape(x_test.shape[0], 784) / 255.

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

tf.random.set_seed(777)

model = Sequential([
Dense(512, input_dim=784, activation='relu'),
Dense(256, activation='relu'),
Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.3, epochs=20, batch_size=200)
print(model.evaluate(x_test, y_test))

import tensorflow as tf
import numpy as np

tf.random.set_seed(777)

data = np.loadtxt('./dataset/dataset/ThoraricSurgery.csv', delimiter=',')
x_data = data[:, 0:17]
y_data = data[:, 17]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy']
)

model.fit(x_data, y_data, epochs=30, batch_size=10)
result = model.evaluate(x_data, y_data)
print()
print(result) # loss value, accuracy
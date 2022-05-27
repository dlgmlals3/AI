import tensorflow as tf
import pandas as pd

tf.random.set_seed(777)

df = pd.read_csv('./dataset/iris.csv',header=None)
#print(df)
dataset = df.values
#print(dataset)

x_data = dataset[:, 0:4].astype(float)
y_data = dataset[:, 4]
#print(y_data)

from sklearn.preprocessing import LabelEncoder
e = LabelEncoder()
y_data = e.fit_transform(y_data)
#print(y_data)
from tensorflow.keras.utils import to_categorical
y_encoded = to_categorical(y_data)
print(y_encoded)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

result = model.fit(x_data, y_encoded, epochs=200, batch_size=10)
print()
print(result.history)

y_loss = result.history['loss']
y_acc = result.history['accuracy']

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.plot(y_loss, 'o', c='red', ms = 2, label = 'loss')
plt.plot(y_acc, 'o', c='blue', ms = 2, label = 'accuracy')
plt.legend(loc = 'best')

plt.show()



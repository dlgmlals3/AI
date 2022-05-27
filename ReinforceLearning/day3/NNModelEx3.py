
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)
df = pd.read_csv('dataset/wine.csv', header=None)
df = df.sample(frac=0.15)

data = df.values
x = data[:, 0:12]
y = data[:, 12]

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy'])

import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

model_dir = './model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

modelpath = './model/{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)
estopcallback = EarlyStopping(monitor='val_loss', patience=100)
result = model.fit(x, y,
                   epochs=300,
                   batch_size=100,
                   validation_split=0.33,
                   callbacks=[checkpointer]
                   )

val_loss = result.history['val_loss']
train_loss = result.history['loss']

val_acc = result.history['val_accuracy']
train_acc = result.history['accuracy']

plt.plot(val_loss, 'o', c='red', ms=3, label='val_loss')
plt.plot(train_loss, 'o', c='blue', ms=3, label='train_loss')
plt.legend(loc='best')
plt.show()

plt.plot(val_acc, 'o', c='red', ms=3, label='val_acc')
plt.plot(train_acc, 'o', c='green', ms=3, label='train_acc')
plt.legend(loc='best')
plt.show()

model.load_weights('./model/0.0900.hdf5')
print(model.evaluate(x, y))

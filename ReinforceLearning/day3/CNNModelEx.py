from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import tensorflow as tf
tf.random.set_seed(777)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# batch, h, w, channel
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255.

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    # CNN 커널 32개, output 채널 32
    Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'), #26 * 26 *32
    Conv2D(64, (3,3), activation='relu'), #24 * 24 * 64
    MaxPooling2D(pool_size=2), # 커널2x2, 스트라이드2x2,  12 * 12 * 64
    # FC Layer
    Flatten(), # 12 by 12 * 64를 --> 직렬화 -->
    Dense(128, activation='relu'), # 9216
    Dropout(0.5), # over feat 막기 위해,
    Dense(10, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    batch_size=200)

print('loss & accuracy:', model.evaluate(x_test, y_test))
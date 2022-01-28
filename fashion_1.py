from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam, Adadelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from google.colab import files
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
# загружаем датасет
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# посмотрим размерности выборок
print(x_train.shape,y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
# посмотрим размерности выборок
print(x_train.shape,y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.astype('float32')/255
x_test_reshaped = x_test_reshaped.astype('float32')/255

class_count = 10
y_train = utils.to_categorical(y_train, class_count)
y_test = utils.to_categorical(y_test, class_count)

# создаём модель из 3-х слоёв:
model = Sequential()
model.add(BatchNormalization(input_shape=(x_train.shape[1],)))
model.add(Dense(800, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(400, activation='relu'))
model.add(Dense(class_count, activation='relu'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train,
          y_train,
          batch_size=1024,
          epochs=15,
          validation_data=(x_train[50000:], y_train[50000:]),
          verbose=1)
print(history.history['loss'][:5])
print('Now we evaluate what we got:')
scores = model.evaluate(x_train, y_train, verbose=1)
print('Total accuracy is: ', scores[1])
print('Loss value is: ', scores[0])
scores_test = model.evaluate(x_test_reshaped, y_test, verbose=1)
print('Accuracy in test selection is: ', scores_test[1])
print('Loss in test selection is: ', scores_test[0])

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

# создаём модель:
# создаём модель:
model_7 = Sequential()
model_7.add(BatchNormalization(input_shape=(x_train.shape[1],)))
model_7.add(Dense(800, activation='relu'))
model_7.add(BatchNormalization())
model_7.add(Dense(400, activation='relu'))
model_7.add(Dense(class_count, activation='sigmoid'))

model_7.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])
history_7 = model_7.fit(x_train,
          y_train,
          batch_size=1024,
          epochs=15,
          validation_data=(x_train[50000:], y_train[50000:]),
          verbose=1)
print()
print('Now we evaluate what we got:')
scores_7 = model_7.evaluate(x_train, y_train, verbose=1)
print('Total accuracy is: ', scores_7[1])
print('Loss value is: ', scores_7[0])
scores_test_7 = model_7.evaluate(x_test_reshaped, y_test, verbose=1)
print('Accuracy in test selection is: ', scores_test_7[1])
print('Loss in test selection is: ', scores_test_7[0])


plt.plot(history_7.history['loss'],
         label='Ошибка на обучающем наборе')

plt.plot(history_7.history['val_loss'],
         label='Ошибка на проверочном наборе')

plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')

plt.legend()

plt.show()
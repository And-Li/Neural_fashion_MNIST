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
print(f'Форма обучающих данных: {x_train.shape} -> {x_train.shape}')
print(f'Форма тестовых данных: {x_test.shape} -> {x_test.shape}')

x_train = x_train.astype('float32')/255
x_test_reshaped = x_test_reshaped.astype('float32')/255

CLASS_COUNT = 10
y_train = utils.to_categorical(y_train, CLASS_COUNT)
y_test = utils.to_categorical(y_test, CLASS_COUNT)
print(y_train.shape)
print(y_train[0])
print(x_train, y_train.shape)

# создаём модель из 3-х слоёв: 28(столько же нейронов, сколько знаков на входе), 14 и 10 на выходе(по кол-ву классов одежды)
model = Sequential()
model.add(Dense(800, input_dim=784, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(CLASS_COUNT, activation='relu'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=128, epochs=15, verbose=1)

n_rec = 389
plt.imshow(x_test[n_rec], cmap='gray')
plt.show()



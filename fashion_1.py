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
x_train = np.reshape(x_train.shape[0], -1)
y_train = np.reshape(y_train.shape[0], -1)
print(x_train.shape, y_train.shape)

'''imgs = np.array([x_train[y_train==i][0] for i in range(10)])
imgs = np.concatenate(imgs, axis=1)
plt.figure()
plt.imshow(imgs, cmap='Greys_r')
plt.grid(False)
plt.axis('off')
plt.show()'''
# создаём модель из 3-х слоёв: 28(столько же нейронов, сколько знаков на входе), 14 и 10 на выходе(по кол-ву классов одежды)
model = Sequential()
model.add(Dense(784, input_dim=x_train.shape[0], activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(10, activation='relu'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.01),
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=28, epochs=100, verbose=1)





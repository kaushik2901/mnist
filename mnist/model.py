import numpy as np
import os
from . import settings
from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical

#load dataset
'''
train_images = np.load('mnist_train_images.npy')
train_labels = np.load('mnist_train_labels.npy')
test_images = np.load('mnist_test_images.npy')
test_labels = np.load('mnist_test_labels.npy')
'''
train_images = np.load(os.path.join(settings.BASE_DIR, 'static/') + 'mnist/mnist_train_images.npy')
train_labels = np.load(os.path.join(settings.BASE_DIR, 'static/') + 'mnist/mnist_train_labels.npy')
test_images = np.load(os.path.join(settings.BASE_DIR, 'static/') + 'mnist/mnist_test_images.npy')
test_labels = np.load(os.path.join(settings.BASE_DIR, 'static/') + 'mnist/mnist_test_labels.npy')

#prepare dataset
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

#change scale and datatype
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

#preparing labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#model
network = Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(512, activation='relu', input_shape=(512,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
network.fit(train_images, train_labels, epochs=30, batch_size=128)

#save model
network.save('mnist_model.h5')



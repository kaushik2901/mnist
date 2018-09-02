import re, io, os, base64
import numpy as np
from PIL import Image
from keras.models import load_model
from django.http import *
from django.shortcuts import render, redirect
from . import settings
import tensorflow as tf

graph = tf.get_default_graph()
network = load_model(os.path.join(settings.BASE_DIR, 'static/') + 'mnist/mnist_model.h5')

'''
from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical

#load dataset
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
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
network.fit(train_images, train_labels, epochs=5, batch_size=128)
'''


def Index(request):
    if request.method == 'POST':
        if 'image' in request.POST:
            image = request.POST['image']
            size = 28, 28

            img = re.search(r'base64,(.*)', image).group(1)
            imbytes = io.BytesIO(base64.b64decode(img))
            image = Image.open(imbytes)
            image = image.resize(size, Image.LANCZOS)
            # image = image.convert('1')

            image = np.array(image)
            image = image[:,:,3]
            image = image.astype('float32') / 255
            image = image.reshape((1, 784))

            with graph.as_default():
                value = network.predict(image)
                value = value.reshape((1, 10))
                value = value.tolist()
                print(value)
                data = {
                    'value': value[0],
                    'predict': value[0].index(max(value[0]))
                }
                return JsonResponse(data)

        data = {
            'value': -99
        }
        return JsonResponse(data)
    return render(request, 'mnist/index.html', {})
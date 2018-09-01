from keras.models import load_model
from keras.utils import to_categorical
import numpy as np

#load dataset
test_images = np.load('mnist_test_images.npy')
test_labels = np.load('mnist_test_labels.npy')

#prepare dataset
test_images = test_images.reshape((10000, 28 * 28))

#change scale and datatype
test_images = test_images.astype('float32') / 255

#preparing labels
test_labels = to_categorical(test_labels)

#load model
network = load_model('mnist_model.h5')

#test model
test_loss, test_accuracy = network.evaluate(test_images, test_labels)

#print result
print("LOSS : " + str(test_loss))
print("ACCURACY : " + str(test_accuracy))

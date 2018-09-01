from keras.datasets import mnist
import numpy as np


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("dataset downloaded")

np.save('mnist_train_images.npy', train_images)
np.save('mnist_train_labels.npy', train_labels)
np.save('mnist_test_images.npy', test_images)
np.save('mnist_test_labels.npy', test_labels)

print("dataset saved")

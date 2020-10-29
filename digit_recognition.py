import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def display_image(image):
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
display_image(test_images[5])
print(test_labels[5])
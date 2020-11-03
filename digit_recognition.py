import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def display_image(image):
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def neural_network(input1, input2, w1, w2, bias):
    return sigmoid(input1 * w1 + input2 * w2 + bias)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def convert_rgb_to_grayscale(images):
    # converts numpy.ndarray of rgb images to numpy.ndarray of grayscale images
    img_rows, img_cols = 28, 28
    images = images.reshape(images.shape[0], img_rows, img_cols, 1)
    images = images.astype(np.float64)
    images /= 255
    return images


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
train_images = convert_rgb_to_grayscale(train_images)
test_images = convert_rgb_to_grayscale(test_images)
pixels = train_images[0].reshape((28, 28))


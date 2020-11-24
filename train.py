import tensorflow as tf
import numpy as np
from neural_network import NeuralNetwork
from matplotlib import pyplot as plt


def display_image(image):
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


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


best = [784, 512, 256, 10]   # [784, 512, 256, 10] - (0.905/ 5) (0.9249/ 10)
nn = NeuralNetwork(best, train=True)
nn.train(train_images, train_labels, epochs=10)


c = 0
for img, label in zip(test_images, test_labels):
    if nn.predict_image(img) == label:
        c += 1

accuracy = c/len(test_labels)
print(accuracy, c)

if accuracy > 0.9269:
    nn.save()
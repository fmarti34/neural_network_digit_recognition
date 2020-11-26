# Handwritten Digit Recognition
![alt text](gif/digit_giphy.gif)

## Neural Network
This is a feed-forward neural network based only in Numpy, which implements and trains a 4 layer neural network.
The neural network defaults to 4 layers, but you are able to train an N layer neural network.
The [MNIST database](http://yann.lecun.com/exdb/mnist/) is used for both training and testing.

## Usage
```bash
pip3 install -r requirements.txt
```
Install the dependencies from requirements.txt
```bash
python3 paint.py
```
This will load model that is saved in /model and allow user to draw digits on UI.

## Train
```bash
python3 train.py
```
This will train and save model to /model. By default this will train a
neural network with an input layer of 786 nodes, two hidden layers of 
512 and 256 nodes respectfully, and one output layer of 10 nodes for
10 epochs. You can adjust the hidden layers and epochs as you see fit.


Now you can start playing around with the code!

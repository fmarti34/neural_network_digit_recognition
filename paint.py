import pygame
import os
from button import Button
from neural_network import NeuralNetwork
import numpy as np
from PIL import Image

pygame.init()

# Initialize neural network
layers = [784, 512, 256, 10]  # Input layer: 784; 2 Hidden layers: 512, 256; 1 output layer: 10
NN = NeuralNetwork(layers)

# get the current working directory
DIR = os.getcwd()

# Image directory
IMAGE_DIR = DIR + '/screenshot.jpg'

# Dimensions
WIDTH, HEIGHT = 280, 280
BTN_WIDTH, BTN_HEIGHT = WIDTH/2, HEIGHT/8

# Colors
BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
LIGHT_BLUE = [160, 210, 219]
MELON = [254, 192, 170]
RED = [237, 69, 81]
LIGHT_GREEN = [145, 226, 131]

# Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Digit Recognition')

# Buttons
CLEAR_BTN = Button(pygame, MELON, RED,  0, HEIGHT-BTN_HEIGHT, BTN_WIDTH, BTN_HEIGHT, 'CLEAR', BLACK, WHITE)
PREDICT_BTN = Button(pygame, LIGHT_BLUE, LIGHT_GREEN, WIDTH/2, HEIGHT-BTN_HEIGHT, BTN_WIDTH, BTN_HEIGHT, 'PREDICT', BLACK, WHITE)


def round_line(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0]+float(i)/distance*dx)
        y = int(start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)


def take_screenshot():
    rect = pygame.Rect(0, 0, WIDTH, HEIGHT-BTN_HEIGHT)
    sub = screen.subsurface(rect)
    pygame.image.save(sub, IMAGE_DIR)


    # im = Image.open(IMAGE_DIR)
    # im_resized = im.resize((28, 28), Image.ANTIALIAS)
    # image = np.asarray(im_resized)
    # print(image.shape)
    # im_resized.save(IMAGE_DIR, "JPEG", optimize=True)

    img = Image.open(IMAGE_DIR)
    img_resized = img.resize((28, 28), Image.ANTIALIAS)
    image = np.mean(img_resized, axis=2)
    img = Image.fromarray(image)
    img = img.convert('L')
    img.save(IMAGE_DIR)


def predict_image():
    with Image.open(IMAGE_DIR) as image:
        image = np.asarray(image)
        print(image.shape)
        print(NN.predict_image(image))
        os.system('say the number is ' + str(NN.predict_image(image)))


def display():
    draw_on = False
    last_pos = [0, 0]
    radius = 10
    while True:
        for event in pygame.event.get():
            pos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # checks if buttons were clicked
                if CLEAR_BTN.is_over(pos):
                    screen.fill(BLACK)
                elif PREDICT_BTN.is_over(pos):
                    take_screenshot()
                    predict_image()
                else:
                    pygame.draw.circle(screen, WHITE, event.pos, radius)
                    draw_on = True
            elif event.type == pygame.MOUSEBUTTONUP:
                draw_on = False
            elif event.type == pygame.MOUSEMOTION:
                if draw_on:
                    pygame.draw.circle(screen, WHITE, event.pos, radius)
                    round_line(screen, WHITE, event.pos, last_pos, radius)
                last_pos = event.pos
            CLEAR_BTN.draw(screen, CLEAR_BTN.is_over(pos))
            PREDICT_BTN.draw(screen, PREDICT_BTN.is_over(pos))
        pygame.display.flip()
display()
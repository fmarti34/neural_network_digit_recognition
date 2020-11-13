import pygame
import os


class Button:
    def __init__(self, color1, color2, x, y, width, height, text, text_color1, text_color2):
        self.color1 = color1
        self.color2 = color2
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.text_color1 = text_color1
        self.text_color2 = text_color2

    def draw(self, win, is_over):
        text_color = self.text_color1
        if is_over:
            pygame.draw.rect(win, self.color2, (self.x, self.y, self.width, self.height), 0)
            text_color = self.text_color2
        else:
            pygame.draw.rect(win, self.color1, (self.x, self.y, self.width, self.height), 0)

        # Adds text to button
        font = pygame.font.Font('freesansbold.ttf', 30)
        text = font.render(self.text, 1, text_color)
        win.blit(text, (self.x + (self.width / 2 - text.get_width() / 2), self.y + (self.height / 2 - text.get_height() / 2)))

    def is_over(self, pos):
        # Pos is the mouse position or a tuple of (x,y) coordinates
        if self.x + self.width > pos[0] > self.x and self.y + self.height > pos[1] > self.y:
            return True
        else:
            return False


pygame.init()

# get the current working directory
DIR = os.getcwd()

# Dimensions
WIDTH, HEIGHT = 600, 600
BTN_WIDTH, BTN_HEIGHT = WIDTH/2, HEIGHT/8

# Colors
BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
LIGHT_BLUE = [160, 210, 219]
MELON = [254, 192, 170]
RED = [237, 69, 81]
LIGHT_GREEN = [145, 226, 131]

# Buttons
CLEAR_BTN = Button(MELON, RED,  0, HEIGHT-BTN_HEIGHT, BTN_WIDTH, BTN_HEIGHT, 'CLEAR', BLACK, WHITE)
PREDICT_BTN = Button(LIGHT_BLUE, LIGHT_GREEN, WIDTH/2, HEIGHT-BTN_HEIGHT, BTN_WIDTH, BTN_HEIGHT, 'PREDICT', BLACK, WHITE)

# Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Digit Recognition')


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
    pygame.image.save(sub, DIR + '/screenshot.jpg')


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
                    # pygame.draw.rect(screen, WHITE, pygame.Rect(0, 0, WIDTH, HEIGHT-BTN_HEIGHT))
                    take_screenshot()
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
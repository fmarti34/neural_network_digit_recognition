import pygame
pygame.init()


class Button:
    def __init__(self, color1, color2, x, y, width, height, text, text_color=None):
        self.color1 = color1
        self.color2 = color2
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.text_color = text_color

    def draw(self, win, is_over):
        if is_over:
            pygame.draw.rect(win, self.color2, (self.x, self.y, self.width, self.height), 0)
        else:
            pygame.draw.rect(win, self.color1, (self.x, self.y, self.width, self.height), 0)

        if self.text != '':
            font = pygame.font.Font('freesansbold.ttf', 30)
            text = font.render(self.text, 1, (0, 0, 0))
            win.blit(text, (self.x + (self.width / 2 - text.get_width() / 2), self.y + (self.height / 2 - text.get_height() / 2)))

    def is_over(self, pos):
        # Pos is the mouse position or a tuple of (x,y) coordinates
        if self.x + self.width > pos[0] > self.x and self.y + self.height > pos[1] > self.y:
            return True
        else:
            return False


WIDTH, HEIGHT = 600, 600
BTN_WIDTH, BTN_HEIGHT = WIDTH/2, HEIGHT/8
LIGHT_BLUE = [160, 210, 219]
MELON = [254, 192, 170]
LIGHT_GREEN = [145, 226, 131]
CLEAR_BTN = Button(MELON, LIGHT_GREEN,  0, HEIGHT-BTN_HEIGHT, BTN_WIDTH, BTN_HEIGHT, 'CLEAR')
PREDICT_BTN = Button(LIGHT_BLUE, LIGHT_GREEN, WIDTH/2, HEIGHT-BTN_HEIGHT, BTN_WIDTH, BTN_HEIGHT, 'PREDICT')

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


def display():
    draw_on = False
    last_pos = (0, 0)
    color = (255, 255, 255)
    radius = 10
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.circle(screen, color, event.pos, radius)
                draw_on = True
            elif event.type == pygame.MOUSEBUTTONUP:
                draw_on = False
            elif event.type == pygame.MOUSEMOTION:
                if draw_on:
                    pygame.draw.circle(screen, color, event.pos, radius)
                    round_line(screen, color, event.pos, last_pos, radius)
                last_pos = event.pos
            pos = pygame.mouse.get_pos()
            CLEAR_BTN.draw(screen, CLEAR_BTN.is_over(pos))
            PREDICT_BTN.draw(screen, PREDICT_BTN.is_over(pos))
        pygame.display.flip()
display()
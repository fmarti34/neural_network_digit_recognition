class Button:
    def __init__(self, pygame, color1, color2, x, y, width, height, text, text_color1, text_color2):
        self.color1 = color1
        self.color2 = color2
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.text_color1 = text_color1
        self.text_color2 = text_color2
        self.pygame = pygame

    def draw(self, win, is_over):
        text_color = self.text_color1
        if is_over:
            self.pygame.draw.rect(win, self.color2, (self.x, self.y, self.width, self.height), 0)
            text_color = self.text_color2
        else:
            self.pygame.draw.rect(win, self.color1, (self.x, self.y, self.width, self.height), 0)

        # Adds text to button
        font = self.pygame.font.Font('freesansbold.ttf', 30)
        text = font.render(self.text, 1, text_color)
        win.blit(text, (self.x + (self.width / 2 - text.get_width() / 2), self.y + (self.height / 2 - text.get_height() / 2)))

    def is_over(self, pos):
        # Pos is the mouse position or a tuple of (x,y) coordinates
        if self.x + self.width > pos[0] > self.x and self.y + self.height > pos[1] > self.y:
            return True
        else:
            return False

class Logger:
    def __init__(self):
        self.xs = []
        self.us = []

    def append_xu(self, x, u):
        self.xs.append(x)
        self.us.append(u)

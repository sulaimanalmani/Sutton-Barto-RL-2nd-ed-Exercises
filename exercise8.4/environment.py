from matplotlib import pyplot as plt
import numpy as np
import random


class Environment:
    def __init__(self, world):
        self.world = world
        self.rows = self.world['rows']
        self.cols = self.world['cols']
        self.start_loc = self.world['start_loc']
        self.end_loc = self.world['end_loc']
        self.blocked_tiles = self.world['blocked_tiles']

        self.area = np.zeros((self.rows, self.cols))
        self.area[self.rows - self.start_loc[0] - 1][self.start_loc[1]] = 1
        self.area[self.rows - self.end_loc[0] - 1][self.end_loc[1]] = 2
        for tile in self.blocked_tiles:
            self.area[self.rows - tile[0] - 1][tile[1]] = 4

        self.current_loc = np.array(self.start_loc)

    def draw_env(self):
        plt.imshow(self.area)
        plt.title("Yellow = After Action, Green = After Wind effect")
        plt.ion()
        plt.show()
        plt.pause(0.1)

    def env_update(self, action, draw):
        end = False
        reward = 0
        moves = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        move_vals = moves[action]

        next_loc = np.array([self.current_loc[0] + move_vals[0],
                             self.current_loc[1] + move_vals[1]])

        next_loc = [max(0, min(self.rows - 1, next_loc[0])), max(0, min(self.cols - 1, next_loc[1]))]
        if next_loc in self.blocked_tiles:
            next_loc = np.array(self.current_loc)

        self.area[self.rows - self.current_loc[0] - 1][self.current_loc[1]] = 0
        self.current_loc = next_loc
        self.area[self.rows - next_loc[0] - 1][next_loc[1]] = 3

        if (next_loc[0] == self.end_loc[0] and next_loc[1] == self.end_loc[1]):
            end = True
            reward = 1

        self.draw_env() if (draw == 1) else 0
        return end, next_loc, reward


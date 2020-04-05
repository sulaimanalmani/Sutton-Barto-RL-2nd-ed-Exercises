from matplotlib import pyplot as plt
import numpy as np
import random


class Car:
    def __init__(self, pos=[0, 0], velocity=[0, 0]):
        self.pos = pos
        self.velocity = velocity

    def gas(self):
        self.pos[0] = self.pos[0] + self.velocity[0]
        self.pos[1] = self.pos[1] + self.velocity[1]

    def accelerate(self, acc):
        updated = 0
        if self.velocity[0] + acc[0] == 0 and self.velocity[1] + acc[1] == 0:
            prefix = np.random.choice([0, 1], 1, p=[0.5, 0.5])
            self.velocity[1] = int(prefix)
            self.velocity[1] = int(1 - prefix)
            updated = 1

        if not updated:
            self.velocity[0] = min(5, max(0, self.velocity[0] + acc[0]))
            self.velocity[1] = min(5, max(0, self.velocity[1] + acc[1]))


class Environment:
    def __init__(self, track, car):
        self.track = track
        self.car = car
        self.rows, self.columns = np.shape(track)
        self.area = np.array(self.track)

    def draw_env(self):
        plt.imshow(self.area)
        plt.ion()
        plt.show()
        plt.pause(0.1)

    def env_acc_car(self, acc):
        self.car.accelerate(acc)

    def env_update(self, action, draw):
        end = False
        tries = 0
        reward = -1
        pedal = [-1, 0, 1]
        toggle = np.random.choice([0, 1], p=[0.1, 0.9])

        self.area[min(self.rows - self.car.pos[0] - 1, self.rows - 1), min(self.car.pos[1], self.columns - 1)] = 4
        self.env_acc_car([pedal[action[0]], pedal[action[1]]]) if toggle else 0

        old_loc = np.array(self.car.pos)
        self.car.gas()
        new_loc = np.array(self.car.pos)

        middle_tiles = [[i, j] for j in np.arange(old_loc[1], new_loc[1] + 1)
                        for i in np.arange(old_loc[0], new_loc[0] + 1)]

        for tile in middle_tiles:
            if tile[0] > self.rows - 1 or tile[1] > self.columns - 1 or \
                    tile[0] < 0 or tile[1] < 0:
                self.draw_env() if draw == 1 else 0
                tries = 1
                self.restart()
                return reward, end, tries

            elif not (self.track[self.rows - tile[0] - 1, tile[1]] == 0 or
                      self.track[self.rows - tile[0] - 1, tile[1]] == 2 or
                      self.track[self.rows - tile[0] - 1, tile[1]] == 3):
                self.draw_env() if draw == 1 else 0
                tries = 1
                self.restart()
                return reward, end, tries

            if self.track[self.rows - tile[0] - 1, tile[1]] == 3:
                end = True
                self.car.pos = tile
                self.draw_env() if draw == 1 else 0
                tries = 1
                return reward, end, tries

        return reward, end, tries

    def restart(self):
        start_cols = np.where(self.track == 2)[1]
        self.car.pos[0] = 0
        self.car.pos[1] = start_cols[random.randrange(len(start_cols))]
        self.car.velocity = [0, 0]
        self.area = np.array(self.track)


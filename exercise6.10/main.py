import numpy as np
from matplotlib import pyplot as plt
import environment


def main():
    world = {
        'rows': 7,
        'cols': 10,
        'start_loc': [3, 0],
        'end_loc': [3, 7],
        'wind_str': [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    }

    actions = [[i, j] for i in range(3) for j in range(3)]
    n_action = np.shape(actions)[0]
    action_values = [[[0 for i in range(n_action)] for j in range(world['cols'])] for k in range(world['rows'])]
    pi = [[[1 / n_action for i in range(n_action)] for j in range(world['cols'])] for k in range(world['rows'])]
    alpha = 0.1
    e = 0.01
    gamma = 1
    max_steps = -1000

    while True:
        print("Episode started -----------")
        env = environment.Environment(world)
        s_t = env.current_loc
        a_t = actions[int(np.random.choice(n_action, 1, p=pi[s_t[0]][s_t[1]][:]))]
        end = False
        total_reward = 0
        draw = 0
        while end == False:
            end, s_t2, r = env.env_update(action=a_t, draw=draw)
            total_reward = total_reward + r
            a_t2 = actions[int(np.random.choice(n_action, 1, p=pi[s_t2[0]][s_t2[1]][:]))]

            action_values[s_t[0]][s_t[1]][actions.index(a_t)] = action_values[s_t[0]][s_t[1]][actions.index(a_t)] + \
                                                                alpha * (r + gamma * action_values[s_t2[0]][s_t2[1]][
                                                                    actions.index(a_t2)] -
                                                                         action_values[s_t[0]][s_t[1]][
                                                                             actions.index(a_t)])
            best_action = np.argmax(action_values[s_t[0]][s_t[1]][:])
            for index, action in enumerate(actions):
                if index == best_action:
                    pi[s_t[0]][s_t[1]][index] = 1 - e + e / n_action
                else:
                    pi[s_t[0]][s_t[1]][index] = e / n_action

            s_t = s_t2
            a_t = a_t2

            if total_reward > -10:
                draw = 1
            else:
                draw = 0

            if end:
                max_steps = total_reward if max_steps < total_reward else max_steps
                print("Rewards" + str(total_reward))
                print("Best Policy max steps = " + str(max_steps))


if __name__ == "__main__":
    main()


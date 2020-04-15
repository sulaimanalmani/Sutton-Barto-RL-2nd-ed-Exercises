import numpy as np
from matplotlib import pyplot as plt
import environment
import random
from mpl_toolkits.mplot3d.axes3d import Axes3D

def main():
    random.seed(1)
    dynaqp = DynaQp()
    random.seed(1)
    dynaq = DynaQ()
    random.seed(1)
    dynaqp_modded = DynaQp_modded()

    plt.plot(dynaqp, 'r', label="DynaQ")
    plt.plot(dynaq, 'b', label="DynaQ+")
    plt.plot(dynaqp_modded, 'g', label="DynaQ+ modded")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("Steps vs Cumulative Reward")
    plt.show()


def DynaQ():
    cum_reward = [0]
    world = {
        'rows': 6,
        'cols': 9,
        'start_loc': [0, 3],
        'end_loc': [5, 8],
        'blocked_tiles': [[2, i] for i in range(1, 9)]
    }

    actions = [i for i in range(4)]
    n_action = np.shape(actions)[0]
    action_values = [[[0 for i in range(n_action)] for j in range(world['cols'])] for k in range(world['rows'])]
    model = [[[[] for i in range(n_action)] for j in range(world['cols'])] for k in range(world['rows'])]
    observed_states_actions = []
    alpha = 0.1
    e = 0.1
    gamma = 1
    steps = 0
    n = 50

    while 0 != 1:
        env = environment.Environment(world)
        s_t = env.current_loc
        a_t = np.argmax(action_values[s_t[0]][s_t[1]][:])
        end = False
        total_reward = 0
        draw = 0
        while end == False:
            if steps > 3000 and env.blocked_tiles[-1] == [2, 8]:
                env.blocked_tiles.pop()
            if steps > 6000:
                return cum_reward
            print(steps) if steps % 100 == 0 else 0
            end, s_t2, r = env.env_update(action=a_t, draw=draw)
            total_reward = total_reward + r
            steps = steps + 1
            a_t2 = np.argmax(action_values[s_t2[0]][s_t2[1]][:])

            action_values[s_t[0]][s_t[1]][actions.index(a_t)] = action_values[s_t[0]][s_t[1]][actions.index(a_t)] + \
                                                                alpha * (r + gamma * action_values[s_t2[0]][s_t2[1]][
                actions.index(a_t2)] -
                                                                         action_values[s_t[0]][s_t[1]][
                                                                             actions.index(a_t)])
            model[s_t[0]][s_t[1]][actions.index(a_t)].append([r, s_t2])
            cum_reward.append(cum_reward[-1] + r)

            if [s_t[0], s_t[1], a_t] not in observed_states_actions:
                observed_states_actions.append([s_t[0], s_t[1], a_t])

            s_t = s_t2

            if (np.argmax(action_values[s_t[0]][s_t[1]][:])) == np.argmin(action_values[s_t[0]][s_t[1]][:]):
                a_t = random.randrange(4)
            else:
                a_t = np.random.choice([random.randrange(4),
                                        np.argmax(action_values[s_t[0]][s_t[1]][:])], p=[e, 1 - e])

            for i in range(n):
                s_t_a_t = random.sample(observed_states_actions, 1)[0]
                s_t_plan = [s_t_a_t[0], s_t_a_t[1]]
                a_t_plan= s_t_a_t[2]
                r_plan, s_t2_plan = model[s_t_plan[0]][s_t_plan[1]][a_t_plan][0]
                a_t2_plan = np.argmax(action_values[s_t2_plan[0]][s_t2_plan[1]][:])

                action_values[s_t_plan[0]][s_t_plan[1]][actions.index(a_t_plan)] = \
                    action_values[s_t_plan[0]][s_t_plan[1]][actions.index(a_t_plan)] + \
                    alpha * (r_plan + gamma *
                             action_values[s_t2_plan[0]][s_t2_plan[1]][
                                 actions.index(a_t2_plan)] -
                             action_values[s_t_plan[0]][s_t_plan[1]][
                                 actions.index(a_t_plan)])

            # if total_reward > -20:
            #     draw = 1
            # else:
            #     draw = 0

def DynaQp():
    cum_reward = [0]
    world = {
        'rows': 6,
        'cols': 9,
        'start_loc': [0, 3],
        'end_loc': [5, 8],
        'blocked_tiles': [[2, i] for i in range(1, 9)]
    }

    actions = [i for i in range(4)]
    n_action = np.shape(actions)[0]
    action_values = [[[0 for i in range(n_action)] for j in range(world['cols'])] for k in range(world['rows'])]
    model = [[[[] for i in range(n_action)] for j in range(world['cols'])] for k in range(world['rows'])]
    alpha = 0.1
    e = 0.1
    gamma = 1
    steps = 0
    n = 50
    k = 1e-15
    tau = np.array([[[0 for i in range(n_action)] for j in range(world['cols'])] for k in range(world['rows'])])
    observed_states_actions = []
    observed_states = []

    while 0 != 1:
        env = environment.Environment(world)
        s_t = env.current_loc
        a_t = np.argmax(action_values[s_t[0]][s_t[1]][:])

        end = False
        total_reward = 0
        draw = 0
        while end == False:
            if steps > 3000 and env.blocked_tiles[-1] == [2, 8]:
                env.blocked_tiles.pop()
            if steps > 6000:
                return cum_reward

            print(steps) if steps % 100 == 0 else 0

            #for val in observed_states_actions:
            #    tau[val[0]][val[1]][val[2]] = tau[val[0]][val[1]][val[2]] + 1
            tau = tau + 1
            tau[s_t[0]][s_t[1]][a_t] = 0
            end, s_t2, r = env.env_update(action=a_t, draw=draw)
            total_reward = total_reward + r
            steps = steps + 1
            a_t2 = np.argmax(action_values[s_t2[0]][s_t2[1]][:])

            action_values[s_t[0]][s_t[1]][actions.index(a_t)] = action_values[s_t[0]][s_t[1]][actions.index(a_t)] + \
                                                                alpha * (r + gamma * action_values[s_t2[0]][s_t2[1]][
                                                                actions.index(a_t2)] - action_values[s_t[0]][s_t[1]][
                                                                actions.index(a_t)])
            model[s_t[0]][s_t[1]][actions.index(a_t)].append([r, s_t2])
            cum_reward.append(cum_reward[-1] + r)

            if [s_t[0], s_t[1], a_t] not in observed_states_actions:
                observed_states_actions.append([s_t[0], s_t[1], a_t])
            if [s_t[0], s_t[1]] not in observed_states:
                observed_states.append([s_t[0], s_t[1]])
            s_t = s_t2

            if (np.argmax(action_values[s_t[0]][s_t[1]][:])) == np.argmin(action_values[s_t[0]][s_t[1]][:]):
                a_t = random.randrange(4)
            else:
                a_t = np.random.choice([random.randrange(4),
                                        np.argmax(action_values[s_t[0]][s_t[1]][:])], p=[e, 1 - e])

            for i in range(n):
                #s_t_plan = random.sample(observed_states, 1)[0]
                s_t_plan = [random.randrange(world['rows']), random.randrange(world['cols'])]
                a_t_plan = random.randrange(4)

                if [s_t_plan[0], s_t_plan[1], a_t_plan] in observed_states_actions:
                    r_plan, s_t2_plan = model[s_t_plan[0]][s_t_plan[1]][a_t_plan][0]
                else:
                    r_plan = 0
                    s_t2_plan = s_t_plan

                r_plan = r_plan + k * np.sqrt(tau[s_t_plan[0]][s_t_plan[1]][a_t_plan])
                a_t2_plan = np.argmax(action_values[s_t2_plan[0]][s_t2_plan[1]][:])

                action_values[s_t_plan[0]][s_t_plan[1]][actions.index(a_t_plan)] = \
                    action_values[s_t_plan[0]][s_t_plan[1]][actions.index(a_t_plan)] + \
                    alpha * (r_plan + gamma *
                             action_values[s_t2_plan[0]][s_t2_plan[1]][
                             actions.index(a_t2_plan)] - action_values[
                             s_t_plan[0]][s_t_plan[1]][actions.index(a_t_plan)])

            # mean_action_values = [[0 for j in range(world['cols'])] for k in range(world['rows'])]
            # for i in range(world['rows']):
            #     for j in range(world['cols']):
            #         mean_action_values[i][j] = np.mean(action_values[i][j][:])
            #
            # x = np.arange(0, world['cols'], 1)
            # y = np.arange(0, world['rows'], 1)
            # xs, ys = np.meshgrid(x, y)
            # zs = np.array(mean_action_values)
            # fig = plt.figure(1)
            # ax = Axes3D(fig)
            # ax.plot_wireframe(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
            # plt.ion()
            # plt.show()
            # plt.pause(0.01)
            # if steps > 3000:
            #     draw = 1
            # else:
            #     draw = 0

def DynaQp_modded():
    cum_reward = [0]
    world = {
        'rows': 6,
        'cols': 9,
        'start_loc': [0, 3],
        'end_loc': [5, 8],
        'blocked_tiles': [[2, i] for i in range(1, 9)]
    }

    actions = [i for i in range(4)]
    n_action = np.shape(actions)[0]
    action_values = [[[0 for i in range(n_action)] for j in range(world['cols'])] for k in range(world['rows'])]
    model = [[[[] for i in range(n_action)] for j in range(world['cols'])] for k in range(world['rows'])]
    alpha = 0.1
    e = 0.1
    gamma = 1
    steps = 0
    n = 50
    k = 1e-15
    tau = np.array([[[0 for i in range(n_action)] for j in range(world['cols'])] for k in range(world['rows'])])
    observed_states_actions = []
    observed_states = []

    while 0 != 1:
        env = environment.Environment(world)
        s_t = env.current_loc
        a_t = np.argmax(action_values[s_t[0]][s_t[1]][:])

        end = False
        total_reward = 0
        draw = 0
        while end == False:
            if steps > 3000 and env.blocked_tiles[-1] == [2, 8]:
                env.blocked_tiles.pop()
            if steps > 6000:
                return cum_reward

            print(steps) if steps % 100 == 0 else 0

            for val in observed_states_actions:
                tau[val[0]][val[1]][val[2]] = tau[val[0]][val[1]][val[2]] + 1
            #tau = tau + 1
            tau[s_t[0]][s_t[1]][a_t] = 0
            end, s_t2, r = env.env_update(action=a_t, draw=draw)
            total_reward = total_reward + r
            steps = steps + 1
            a_t2 = np.argmax(action_values[s_t2[0]][s_t2[1]][:])

            action_values[s_t[0]][s_t[1]][actions.index(a_t)] = action_values[s_t[0]][s_t[1]][actions.index(a_t)] + \
                                                                alpha * (r + gamma * action_values[s_t2[0]][s_t2[1]][
                                                                actions.index(a_t2)] - action_values[s_t[0]][s_t[1]][
                                                                actions.index(a_t)])
            model[s_t[0]][s_t[1]][actions.index(a_t)].append([r, s_t2])
            cum_reward.append(cum_reward[-1] + r)

            if [s_t[0], s_t[1], a_t] not in observed_states_actions:
                observed_states_actions.append([s_t[0], s_t[1], a_t])
            if [s_t[0], s_t[1]] not in observed_states:
                observed_states.append([s_t[0], s_t[1]])
            s_t = s_t2

            if np.random.choice([0, 1], 1, p=[e, 1-e]):
                a_t = random.randrange(4)
            else:
                modded_action_values = action_values[s_t[0]][s_t[1]][:] + k * np.sqrt(tau[s_t[0]][s_t[1]][:])
                if np.argmax(modded_action_values) == np.argmin(modded_action_values):
                    a_t = random.randrange(4)
                else:
                    a_t = np.argmax(modded_action_values)

            for i in range(n):
                s_t_a_t = random.sample(observed_states_actions, 1)[0]
                s_t_plan = [s_t_a_t[0], s_t_a_t[1]]
                a_t_plan = s_t_a_t[2]
                r_plan, s_t2_plan = model[s_t_plan[0]][s_t_plan[1]][a_t_plan][0]
                a_t2_plan = np.argmax(action_values[s_t2_plan[0]][s_t2_plan[1]][:])

                action_values[s_t_plan[0]][s_t_plan[1]][actions.index(a_t_plan)] = \
                    action_values[s_t_plan[0]][s_t_plan[1]][actions.index(a_t_plan)] + \
                    alpha * (r_plan + gamma *
                             action_values[s_t2_plan[0]][s_t2_plan[1]][
                                 actions.index(a_t2_plan)] -
                             action_values[s_t_plan[0]][s_t_plan[1]][
                                 actions.index(a_t_plan)])

if __name__ == "__main__":
    main()


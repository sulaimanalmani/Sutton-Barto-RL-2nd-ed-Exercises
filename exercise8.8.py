import numpy as np
import random
from matplotlib import pyplot as plt


def policy_eval(action_values, rewards, connections, b):
    s_val = []
    episode = 0
    while episode < 100:
        s_t = 0
        s_val_temp = 0
        end = False
        s_t = 0
        e = 0.1
        while not end:
            a_t = random.randrange(2)
            if not all(x == action_values[s_t, 0] for x in action_values[s_t, :]) \
                    and np.random.choice([0, 1], p=[e, 1 - e]) == 1:
                a_t = np.argmax(action_values[s_t, :])

            p = [0.9 / b for i in range(b)]
            p.append(0.1)
            n = np.random.choice(np.arange(b + 1), p=p)

            if n != b:
                s_t_next = connections[s_t, a_t, n]
                s_val_temp = s_val_temp + rewards[s_t, a_t, n]
                s_t = s_t_next
            else:
                end = True
                episode = episode + 1
                s_val_temp = s_val_temp + rewards[s_t, a_t, n]
                s_val.append(s_val_temp)
    return np.mean(s_val)


# def policy_eval2(action_values, rewards, connections, b, n_state=10000, e=0.1):
#     delta = 1
#     gamma = 1
#     theta = 0.1
#     state_values = np.zeros(shape=n_state)
#
#     while delta > theta:
#         delta = 0
#         for s in range(n_state):
#             v = state_values[s]
#
#             temp_eval = 0
#             p = 1 - e + e / 2
#             a = np.argmax(action_values[s, :])
#             s_next = connections[s, a, :]
#             r_eval = rewards[s, a, :]
#             temp_eval = temp_eval + p * (
#                 np.sum((0.9 / b * (r_eval[:-1][0] + gamma * state_values[s_next])) + 0.1 * (r_eval[-1])))
#
#             p = e / 2
#             a = (a + 1) % 2
#             s_next = connections[s, a, :]
#             r_eval = rewards[s, a, :]
#             temp_eval = temp_eval + p * (
#                 np.sum((0.9 / b * (r_eval[:-1][0] + gamma * state_values[s_next])) + 0.1 * (r_eval[-1])))
#             state_values[s] = float(temp_eval)
#
#             delta = max(delta, abs(v - state_values[s]))
#     return state_values[0]


def policy_sampling(rewards, connections, iters, b, n_state):
    s0_value = []
    n_action = 2
    e = 0.1
    gamma = 1
    action_values = np.array([[0 for i in range(n_action)] for j in range(n_state)], dtype=np.float)

    step = 0
    while step <= iters:
        s_t = 0
        end = False
        while not end:
            if step % 10000 == 0:
                print("step = " + str(step))
            s0_value.append(policy_eval(action_values, rewards, connections, b)) if step % 1000 == 0 else 0
            if step > iters + 1:
                break

            a_t = random.randrange(2)
            if not all(x == action_values[s_t, 0] for x in action_values[s_t, :]) \
                    and np.random.choice([0, 1], p=[e, 1 - e]) == 1:
                a_t = np.argmax(action_values[s_t, :])

            s_t_next = random.sample(connections[s_t, a_t, :].tolist(), 1)[0]
            temp = 0

            if np.random.choice([0, 1], p=[0.1, 0.9]) == 0:
                end = True

            step = step + 1

            for i in range(b):
                temp_s_t_next = connections[s_t, a_t, i]
                r = rewards[s_t, a_t, i]
                temp = temp + 0.9 * (1 / b) * (r + gamma * np.max(action_values[temp_s_t_next, :]))
            r = rewards[s_t, a_t, -1]
            temp = temp + 0.1 * r
            action_values[s_t, a_t] = float(temp)

            s_t = s_t_next
    return s0_value


def uniform_dist(rewards, connections, iters, b, n_state):
    s0_value = []
    n_action = 2
    e = 0.1
    gamma = 1
    action_values = np.array([[0 for i in range(n_action)] for j in range(n_state)], dtype=np.float)

    step = 0
    while step <= iters:
        s_t = 0
        end = False
        while not end:
            if step > iters + 1:
                break

            for a_t in range(2):
                if step % 10000 == 0:
                    print("step = " + str(step))
                s0_value.append(policy_eval(action_values, rewards, connections, b)) if step % 1000 == 0 else 0
                temp = 0
                for i in range(b):
                    temp_s_t_next = connections[s_t, a_t, i]
                    r = rewards[s_t, a_t, i]
                    temp = temp + 0.9 * (1 / b) * (r + gamma * np.max(action_values[temp_s_t_next, :]))
                r = rewards[s_t, a_t, -1]
                temp = temp + 0.1 * r
                action_values[s_t, a_t] = float(temp)
                step = step + 1

            s_t = (s_t + 1) % n_state

    return s0_value


def main():
    n_state = 10000
    n_action = 2  # Don't change this value.
    b = 3
    iters = 200000
    sample_dist = np.zeros(shape=int(iters / 1000 + 1))
    uniform_distrib = np.zeros(shape=int(iters / 1000 + 1))

    for i in range(200):
        print("Sample Task no:", i)
        rewards = np.array([[[np.random.normal(0, 1, 1)[0] for _ in range(b + 1)] for _ in range(n_action)
                             ] for _ in range(n_state)])
        connections = np.array(
            [[[random.sample(range(n_state), b)][0] for i in range(n_action)] for j in range(n_state)])

        sample_dist = np.vstack((sample_dist, policy_sampling(rewards, connections, iters, b, n_state)))
        uniform_distrib = np.vstack((uniform_distrib, uniform_dist(rewards, connections, iters, b, n_state)))

        plt.clf()
        plt.plot(np.mean(sample_dist[1:, :], axis=0), 'b', label='Policy Sampling')
        plt.plot(np.mean(uniform_distrib[1:, :], axis=0), 'r', label='Uniform Distribution')
        plt.title("Uniform Distribution vs Policy Sampling (b=" + str(b) + ", n_states=" + str(n_state) + ")")
        plt.xlabel("Expected Updates x1e3")
        plt.ylabel("Initial State Value")
        plt.legend()
        plt.ion()
        plt.show()
        plt.pause(0.01)


if __name__ == "__main__":
    main()


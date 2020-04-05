import environment
import tracks
from matplotlib import pyplot as plt
import numpy as np

def main():
    race_track = tracks.TRACKS['RACETRACK_3']
    rows = int(np.shape(race_track)[0])
    cols = int(np.shape(race_track)[1])
    Actions = [[i, j] for i in range(3) for j in range(3)]
    States = [[i, j] for i in range(rows) for j in range(cols)]
    n_Action = np.shape(Actions)[0]
    n_states = (race_track.shape[0]*race_track.shape[1])

    #Pi = np.full((n_states, n_Action), 1 / n_Action)
    Pi = [[[1 / n_Action for i in range(n_Action)] for j in range(race_track.shape[1])] for k in range(race_track.shape[0])]
    Action_Values = [[[0 for i in range(n_Action)] for j in range(race_track.shape[1])] for k in range(race_track.shape[0])]
    Returns = [[[[] for i in range(n_Action)] for j in range(race_track.shape[1])] for k in range(race_track.shape[0])]


    gamma = 0.9
    e = 0.25
    car = environment.Car()
    env = environment.Environment(race_track, car)
    env.restart()
    episode_number = 0
    while True:
        print("Generating Episode #" + str(episode_number))
        episode_number = episode_number + 1
        episode_end = False
        episode_steps = 0
        states_faced = []
        actions_faced = []
        rewards_faced = []
        env.restart()
        draw = 0
        retries = 0
        while episode_end is False:
            #print(" Episode step #" + str(episode_steps))
            current_state = [env.car.pos[0], env.car.pos[1]]
            current_action = Actions[int(np.random.choice(n_Action, 1, p=Pi[current_state[0]][current_state[1]][:]))]

            states_faced.append(current_state)
            actions_faced.append(current_action)

            next_reward, episode_end, retry = env.env_update(action=current_action, draw=draw)
            retries = retries + retry
            rewards_faced.append(next_reward)
            episode_steps = episode_steps + 1

            # print(current_action)
            draw = 0
            if episode_number % 100 == 0 and episode_number > 1:
                draw = 1
        print("restarts before finish = " + str(retries))
        print("------------------")
        StAt_pairs = [[states_faced[:][i], actions_faced[:][i]] for i, val in enumerate(states_faced[:])]
        g = 0
        for step in np.arange(-1, -1*(episode_steps+1), -1):
            g = gamma * g + rewards_faced[step]
            if [states_faced[step], actions_faced[step]] not in StAt_pairs[:step]:

                S_t = states_faced[step]
                A_t = actions_faced[step]
                Returns[S_t[0]][S_t[1]][Actions.index(A_t)].append(g)
                Action_Values[S_t[0]][S_t[1]][Actions.index(A_t)] = np.mean(Returns[S_t[0]][S_t[1]][Actions.index(A_t)])

                best_action = np.argmax(Action_Values[S_t[0]][S_t[1]][:])
                for index, action in enumerate(Actions):
                    if index == best_action:
                        Pi[S_t[0]][S_t[1]][index] = 1 - e + (e / n_Action)
                    else:
                        Pi[S_t[0]][S_t[1]][index] = e / n_Action

if __name__ == "__main__":
    main()


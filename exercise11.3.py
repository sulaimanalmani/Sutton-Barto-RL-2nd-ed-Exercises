import numpy as np
import random
from matplotlib import pyplot as plt
def main():

    state_feature_vectors = np.array([[2,0,0,0,0,0,0,1],
                       [0,2,0,0,0,0,0,1],
                       [0,0,2,0,0,0,0,1],
                       [0,0,0,2,0,0,0,1],
                       [0,0,0,0,2,0,0,1],
                       [0,0,0,0,0,2,0,1],
                       [0,0,0,0,0,0,1,2]])
    w = np.array([1,1,1,1,1,1,10,1,1])
    alpha = 0.01
    gamma = 0.99
    behav_trans_probs = [6/7,1/7]
    w_history = w
    s = random.randrange(7)
    steps = 0

    while True:
        steps = steps + 1
        if steps == 200:
            break

        a = 6 if np.random.choice([0, 1], p=[behav_trans_probs[0], behav_trans_probs[1]]) \
            else random.randrange(6)

        q_s_a = np.dot(np.concatenate((state_feature_vectors[s],[a])),w)
        s_dash = a
        q_s_dash_a = 0
        for i in range(7):
            temp = np.dot(np.concatenate((state_feature_vectors[s_dash],[i])),w)
            q_s_dash_a = max(q_s_dash_a,temp)

        delta = 0 + gamma * q_s_dash_a - q_s_a
        w = w + np.multiply(alpha * delta, np.concatenate((state_feature_vectors[s],[a])))
        w_history = np.vstack((w_history,w))

        s = s_dash

    for i in range(8):
        plt.plot(w_history[:,i])
    plt.show()
    plt.title('Biard\'s Counter Example')
    plt.xlabel('Update Step')
    plt.ylabel('Weight Values')
if __name__ == "__main__":
    main()

import numpy as np
from matplotlib import pyplot as plt
import random

gamma = 0.9
goal = 100
ph = 0.55

States = [i for i in range(goal+1)]
Values = [0 for i in States]
Values[-1] = 1
Actions = [np.arange(min(i+1, goal-i+1)) for i in States]
Pi = np.zeros(shape=goal)
theta = 1e-25
delta = 1

while(delta>theta):
    delta = 0;
    for state in range(1, goal):
        v = Values[state]
        max_V = -0.1
        for action in Actions[state]:
            Vs_temp = (ph * gamma * Values[int(state+action)]) + ((1-ph) * Values[int(state-action)])
            if Vs_temp >= max_V:
                max_V = Vs_temp
                Values[state] = Vs_temp
        delta = max(delta, abs(Values[state] - v))


for state in range(1, goal):
    max_V = -0.1
    for action in Actions[state]:
        Vs_temp = (ph * gamma * gamma * Values[int(state+action)]) + ((1-ph) * gamma * Values[int(state-action)])
        if Vs_temp >= max_V:
            Pi[state] = action

plt.subplot(121)
plt.plot(Pi, 'b.')
plt.title("Policy")
plt.xlabel("Current $")
plt.ylabel("Possible Optimal $ Bet")

plt.subplot(122)
plt.plot(Values)
plt.title("Value")
plt.xlabel("State")
plt.ylabel("Value of State")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plt.show()



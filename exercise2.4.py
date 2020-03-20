import numpy as np
import matplotlib.pyplot as plt
from random import randrange, choice

from random import randrange

mu, sigma = 0, 1 # mean and standard deviation of action values
Action_val = np.random.normal(mu, sigma, 10);
ep = 0.1; iters=1000; bandits = 1000;
Optimal_act_idx =  np.where(Action_val == np.amax(Action_val))[0][0];


# For alpha = 1/k
bandit_Rs = [];
bandit_Opt_Acts = [];

for b in range(bandits):
    R_vals = [];
    Q_vals = [5 for i in range(10)];
    k = [0 for i in range(10)];
    Opt_act_pct = [];
    for n in range(iters):
        rand_prob = randrange(1001)/1000;
        if rand_prob<(1-ep):
            #Max Action Value
            At = np.where(Q_vals == np.amax(Q_vals))[0];

            #Symmetry breaking
            rand_At = randrange(len(At));
            At_idx = At[rand_At];

            #Updating Action Values estimate
            k[At_idx] = k[At_idx] + 1;
            R = Action_val[At_idx] + np.random.normal(1, 1, 1)[0];
            Q_vals[At_idx] = Q_vals[At_idx] + ((1/k[At_idx]) * (R - Q_vals[At_idx]));
            R_vals.append(R);
        else:
            #Radom Action
            At_idx = randrange(10);

            #Updating Action Values estimate
            k[At_idx] = k[At_idx] + 1;
            R = Action_val[At_idx] + np.random.normal(1, 1, 1)[0];
            Q_vals[At_idx] = Q_vals[At_idx] + ((1/k[At_idx]) * (R - Q_vals[At_idx]));
            R_vals.append(R);
        Opt_act_pct.append(k[Optimal_act_idx]/(n+1));
    bandit_Opt_Acts.append(Opt_act_pct);
    bandit_Rs.append(R_vals);

bandit_average = np.divide([ sum(x) for x in zip(*bandit_Rs) ],bandits);
OptAct_average = np.divide([ sum(x) for x in zip(*bandit_Opt_Acts) ],bandits);
fig, axs = plt.subplots(2)
#axs[0].suptitle('10-armed bandits average(Gaussian Dist Rewards mu=0, signa=1)')
axs[0].plot(np.arange(len(bandit_average)),bandit_average,'r',label="alpha = 1/k");
axs[1].plot(np.arange(len(OptAct_average)),OptAct_average,'r',label="alpha = 1/k");


# For alpha = 1/k
bandit_Rs = [];
bandit_Opt_Acts = [];

for b in range(bandits):
    R_vals = [];
    Q_vals = [5 for i in range(10)];
    k = [0 for i in range(10)];
    Opt_act_pct = [];
    for n in range(iters):
        rand_prob = randrange(1001)/1000;
        if rand_prob<(1-ep):
            #Max Action Value
            At = np.where(Q_vals == np.amax(Q_vals))[0];

            #Symmetry breaking
            rand_At = randrange(len(At));
            At_idx = At[rand_At];

            #Updating Action Values estimate
            k[At_idx] = k[At_idx] + 1;
            R = Action_val[At_idx] + np.random.normal(1, 1, 1)[0];
            Q_vals[At_idx] = Q_vals[At_idx] + (0.1 * (R - Q_vals[At_idx]));
            R_vals.append(R);

        else:
            #Radom Action
            At_idx = randrange(10);

            #Updating Action Values estimate
            k[At_idx] = k[At_idx] + 1;
            R = Action_val[At_idx] + np.random.normal(1, 1, 1)[0];
            Q_vals[At_idx] = Q_vals[At_idx] + (0.1 * (R - Q_vals[At_idx]));
            R_vals.append(R);
        Opt_act_pct.append(k[Optimal_act_idx]/(n+1));
    bandit_Opt_Acts.append(Opt_act_pct);
    bandit_Rs.append(R_vals);

bandit_average = np.divide([ sum(x) for x in zip(*bandit_Rs) ],bandits);
OptAct_average = np.divide([ sum(x) for x in zip(*bandit_Opt_Acts) ],bandits);

axs[0].plot(np.arange(len(bandit_average)),bandit_average,'g',label="alpha = 0.1");
axs[1].plot(np.arange(len(OptAct_average)),OptAct_average,'g',label="alpha = 0.1");
plt.show();
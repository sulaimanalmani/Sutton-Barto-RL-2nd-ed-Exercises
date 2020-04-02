import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

# Initialization
Values = np.zeros(shape=(20, 20))
Pi = np.zeros(shape=(20,20))
Actions = np.arange(-5,6,1);
States = [[i,j] for i in range(20) for j in range(20)];
gamma = 0.9;
#Initializing Return and Request probabilities for Loc 1 and Loc 2
poission_len = 10;
e = 2.718282;
poission_locs = np.arange(poission_len);
loc1_P_req = np.multiply(np.divide(3**poission_locs, factorial(poission_locs)), e**(-1*3));
loc1_P_ret = np.multiply(np.divide(3**poission_locs, factorial(poission_locs)), e**(-1*3));
loc2_P_req = np.multiply(np.divide(4**poission_locs, factorial(poission_locs)), e**(-1*4));
loc2_P_ret = np.multiply(np.divide(2**poission_locs, factorial(poission_locs)), e**(-1*2));

P_stable = False;
iters = 0;
while(P_stable==False):
    iters = iters+1;
    if iters > 10:
        break;
    delta = 6;
    while(delta > 5):
        delta = 0;
        #Policy Evaluation

        for state in States:
            v = Values[state[0], state[1]];
            Values[state[0], state[1]] = 0;
            pi_s = Pi[state[0],state[1]];
            #Going over all the possible return and requests

            for loc1_req in range(poission_len):
                for loc1_ret in range(poission_len):
                    for loc2_req in range(poission_len):
                        for loc2_ret in range(poission_len):

                            next_state = [state[0]- pi_s - loc1_req + loc1_ret,state[1] + pi_s - loc2_req + loc2_ret];
                            #if ~((next_state[0]>-1) and (next_state[0]<20) and (next_state[1]>-1) and (next_state[1]<20)):
                            #    continue;
                            loc1_req_adj = loc1_req;
                            loc2_req_adj = loc2_req;

                            if next_state[0] < 0:
                                if state[0] - pi_s < 0:
                                    continue;
                                else:
                                    loc1_req_adj = int(state[0] - pi_s);
                                    next_state[0] = 0;
                            elif next_state[0] > 19:
                                next_state[0] = 19;

                            if next_state[1] < 0:
                                if state[1] + pi_s < 0:
                                    continue;
                                else:
                                    loc2_req_adj = int(state[1] + pi_s);
                                    next_state[1] = 0;
                            elif next_state[1] > 19:
                                next_state[1] = 19;

                            p_s_a_ns = loc1_P_req[loc1_req] * loc1_P_ret[loc1_ret] * loc2_P_req[loc2_req] * loc2_P_ret[loc2_ret];
                            Values[int(state[0]),int(state[1])] = Values[state[0],state[1]] + p_s_a_ns * ((loc1_req_adj+loc2_req_adj)*10 + (-2)*(abs(pi_s)) + gamma*Values[int(next_state[0]),int(next_state[1])]);

            delta = max(delta, abs(v - Values[state[0],state[1]]));

        print(delta)

    # Policy Iteration
    P_stable = True;
    Pi_temp = Pi;
    for state in States:
        pi_s = Pi[state[0],state[1]];
        max_action_value = 0;
        for action in Actions:
            action_value = 0;
            for loc1_req in range(poission_len):
                for loc1_ret in range(poission_len):
                    for loc2_req in range(poission_len):
                        for loc2_ret in range(poission_len):

                            next_state = [state[0]- action - loc1_req + loc1_ret,state[1] + action - loc2_req + loc2_ret];
                            #if ~((next_state[0]>-1) and (next_state[0]<20) and (next_state[1]>-1) and (next_state[1]<20)):
                            #    continue;
                            loc1_req_adj = loc1_req;
                            loc2_req_adj = loc2_req;

                            if next_state[0]<0:
                                if state[0] - action < 0:
                                    continue;
                                else:
                                    loc1_req_adj = int(state[0] - action);
                                    next_state[0] = 0;
                            elif next_state[0]>19:
                                next_state[0] = 19;

                            if next_state[1]<0:
                                if state[1] + action < 0:
                                   continue;
                                else:
                                    loc2_req_adj = int(state[1] + action);
                                    next_state[1] = 0;
                            elif next_state[1]>19:
                                next_state[1]=19;

                            p_s_a_ns = loc1_P_req[loc1_req] * loc1_P_ret[loc1_ret] * loc2_P_req[loc2_req] * loc2_P_ret[loc2_ret];
                            action_value = action_value + p_s_a_ns * ((loc1_req_adj+loc2_req_adj)*10 + (-2)*(abs(action)) + gamma*Values[int(next_state[0]),int(next_state[1])]);

            if (action_value > max_action_value) :
                max_action_value = action_value;
                Pi[state[0],state[1]] = action;
        if pi_s != Pi[state[0],state[1]]:
            print("state: [" + str(state[0])+ "," + str(state[1])+"]" )
            print("Changed from: "+str(pi_s)+" to: "+str(Pi[state[0],state[1]]))
            P_stable = False;

    plt.figure(1)
    plt.clf()
    plt.subplot (121)
    plt.title("Policy Pi(s)")
    plt.xlabel("Cars in 1st parking lot")
    plt.ylabel("Cars in 2nd parking lot")
    plt.imshow(Pi);
    plt.colorbar()

    plt.subplot(122)
    plt.title("Values function V(s)")
    plt.xlabel("Cars in 1st parking lot")
    plt.xlabel("Cars in 2nd parking lot")
    plt.imshow(Values);
    plt.colorbar()
    plt.ion()
    if P_stable or iter==10:
        plt.ioff();

    plt.show()
    plt.pause(0.001)
    print(P_stable)

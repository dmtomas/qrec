import numpy as np
from .utils import *

def updates(indb, n, g, r, q0, q1, n0, n1, rewards, lr=0.01):
    q1[indb, n, g]+= lr*(r - q1[indb, n, g])
    q0[indb]+= lr*np.max([q1[indb, n, g] for g in [0,1]] - q0[indb])
    n0[indb]+= 2 * r - 1
    n1[indb,n,g]+= 2 * r - 1
    return q0, q1, n0, n1

def Update_reload(n0, n1, epsilon, change):
    epsilon += (1-epsilon) * np.abs(change)
    for i in range(0, len(n1)):
        n0[i] /= 1 + 1 * np.abs(change)
        for j in range(0, len(n1[i])):
            for k in range(0, len(n1[i][j])):
                n1[i, j, k] /= 1 + 1 * np.abs(change)
    return n0, n1, epsilon

def Experiment_run(details, epsilon0, N, q0, q1, n0, n1, betas_grid, alpha, N0=0, delta1= 50):
    start = time.time()
    mean_rew = 0
    points = [0, 0]
    epsilon = epsilon0
    means = []
    for experiment in range(N0, N0 + N):
        epsilon = epsilon0 + np.exp(-0.5 * experiment / N)
        if experiment%int(N/10)==0:
            print(experiment)
        hidden_phase = np.random.choice([0,1])
        indb, b = ep_greedy(q0, betas_grid, ep=epsilon)
        n = give_outcome(hidden_phase, b, alpha=alpha)
        indg, g = ep_greedy(q1[indb,n,:], [0,1], ep=epsilon)
        r = give_reward(g,hidden_phase)

        means.append(r)
        if len(means) > delta1:
            means.pop(0)
        if experiment % delta1 == 0:
            points[0] = points[1]
            mean_rew = 0
            for i in range(len(means)):
                mean_rew += means[i] / delta1 
            points[1] = mean_rew
            mean_deriv = (points[1] - points[0]) / delta1
            if mean_deriv < 0:
                n0, n1, epsilon = Update_reload(n0, n1, epsilon, mean_deriv)



        q0, q1, n0, n1 = updates(indb, n, g, r, q0, q1, n0, n1, r)
        details["experience"].append([b,n,g,r])
        details["Ps_greedy"].append(Psq(q0,q1,betas_grid,alpha=alpha))
    details["tables"] = [q0,q1,n0,n1]
    end = time.time() - start
    details["total_time"] = end
    return details
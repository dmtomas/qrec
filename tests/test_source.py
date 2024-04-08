import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
from qrec.Stage_run import PhotonSource
from qrec.utils import detection_state_probability, give_outcome


def test_give_outcome():
    alpha=1.5
    beta=0.
    lambd = 0
    phases = np.random.random(100)
    p_1 = detection_state_probability(-alpha, beta, lambd, 1)
    outcomes =[give_outcome(phase, beta, alpha,lambd=0.) for phase in phases]
    sq = (p_1*(1-p_1))
    relative_error = (p_1 - np.average(outcomes))**2/max(sq,1)
    assert relative_error <.01


    alpha=1.5
    beta=1.5
    lambd = 0
    phases = np.random.random(100)
    p_1 = detection_state_probability(-alpha, beta, lambd, 1)
    outcomes =[give_outcome(phase, beta, alpha,lambd=0.) for phase in phases]
    sq = (p_1*(1-p_1))
    relative_error = (p_1 - np.average(outcomes))**2/max(sq,1)
    assert relative_error <.01




def test_source():
    bias = 0
    lambd = 0.
    alpha = 1.5
    source = PhotonSource(alpha=alpha, lambd=lambd,  bias=bias)

    for beta, case in [(0., "no tunning"),(alpha, "tunned"), (-alpha,"opposite")]:
        source.beta = beta
        msg, outcomes =  source.training_signal(10000)
    
        mean_msg = np.average(msg)
        mean_outcomes = np.average(outcomes)
        corr = np.average([m*o for m,o in zip(msg, outcomes)]) - mean_msg*mean_outcomes
        
        prob_d =  sum(detection_state_probability(alpha_phase, source.beta, lambd, 1)*p
                      for alpha_phase, p in zip([alpha,-alpha],[.5-bias, .5+bias]))

        expected_corr = detection_state_probability(-alpha, source.beta, lambd, 1) * (.5+bias) - (.5+bias)*prob_d 
        
        assert abs(mean_msg -.5+bias) <.02, case+": The average of msg "+str(mean_msg)+"must be close to " + str(.5-bias)
        assert abs(mean_outcomes -prob_d) <.02, case + ": the average of the outcomes "+str(mean_outcomes)+"must be close to "+str(prob_d)
        assert abs(corr-expected_corr) <.02, case + ": the correlation " + str(corr) + "must be close to 0."
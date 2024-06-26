"""
Model 50
"""

import os
import pickle
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from qrec.stage_run import run_experiment
from qrec.qlearning import Hyperparameters
from qrec.utils import  bayes_decision_error_probability, model_aware_optimal


EXPERIMENT_INDEX = 2
EXPERIMENT_PATH = "experiments/stability_test/"


def read_cmd_args():
    """read the command line arguments"""
    parser = ArgumentParser(description="Experiment 1", add_help=False)
    parser.add_argument(
        "--help", "-h", help="show this help message and exit", action="help"
    )
    parser.add_argument(
        "--rounds",
        "-r",
        help="Number of rounds",
        type=int,
        dest="amount_vals",
        default=int(10),
        metavar="AMOUNT_VALS",
    )

    parser.add_argument(
        "--training-size",
        "-N",
        help="Size of the training set",
        type=int,
        dest="training_size",
        default=int(5e4),
        metavar="TRAINING_SIZE",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        help="Nominal amplitude of the coherent state",
        type=float,
        dest="alpha",
        default=0.25,
        metavar="ALPHA",
    )
    parser.add_argument(
        "--beta-steps",
        "-g",
        help=("Granularity of the beta parameter space."),
        type=int,
        default=10,
        dest="beta_steps",
        metavar="BETA_STEPS",
    )
    parser.add_argument(
        "--dispersion",
        "-d",
        help=("Granularity of the beta parameter space."),
        type=float,
        default=1.0,
        dest="dispersion",
        metavar="DISPERSION",
    )
    parser.add_argument(
        "--epsilon",
        "-e",
        help=("Value of the epsilon parameter."),
        type=float,
        dest="epsilon",
        default=1.0,
        metavar="EPSILON",
    )

    parser.add_argument(
        "--lambda",
        "-l",
        help=("Amout of noise in the source."),
        type=float,
        default=0.75,
        dest="lambd",
        metavar="LAMBDA",
    )
    parser.add_argument(
        "--noise-type",
        "-t",
        help=("Type of noise introduced."),
        type=int,
        default=1,
        dest="noise_type",
        metavar="NOISE_TYPE",
    )
    parser.add_argument(
        "--random-seed",
        "-s",
        help=("Seed of the RNG."),
        type=int,
        dest="random_seed",
        default=0,
        metavar="RANDOM_SEED",
    )
    return parser.parse_args()


def make_plots(
    q0s, rts, betas, alpha, lambd, noise_type, betas_grid, beta_star, p_star
):
    """Make the plots"""
    lw = 4
    s= 20
    a = .8
    ct = "red"
    ca = "blue"
    ao = .5
    means_rts = [0 for i in range(len(rts[0]))]
    means_q0s = [0 for i in range(len(q0s[0]))]
    means_betas = [0 for i in range(len(betas[0]))]

    for i, q0s_i in enumerate(q0s):
        for j, q0s_ij in enumerate(q0s_i):
            means_q0s[j] += q0s_ij / len(q0s)

    for i, rts_i in enumerate(rts):
        for j, rts_ij in enumerate(rts_i):
            means_rts[j] += rts_ij / len(rts)

    for i, betas_i in enumerate(betas):
        for j, betas_ij in enumerate(betas_i):
            means_betas[j] += betas_ij / len(betas)

    f, ax = plt.subplots(1, 2)

    disp_q0s = [[], []]
    for i in range(len(q0s[0])):
        # TODO: maybe use reshape?
        vals = np.array([q0s_j[i] for j, q0s_j in enumerate(q0s)])
        disp_q0s[0].append(vals.min())
        disp_q0s[1].append(vals.max())

    ax[0].fill_between(betas_grid, disp_q0s[0], disp_q0s[1], alpha=0.3, color="orange")
    ax[0].plot(betas_grid, means_q0s, color=ct, linewidth=lw, label=r"$\langle agent\rangle$")
    ax[0].plot(
        betas_grid,
        [
            1
            - bayes_decision_error_probability(
                b, alpha=alpha, noise_val=lambd, noise_type=noise_type
            )
            for b in betas_grid
        ],
        label=r'$S_\mathcal{E} (\theta)$', 
        color="black", 
        linewidth=lw
    )
    ax[0].plot(
        betas_grid,
        [
            1 - bayes_decision_error_probability(b, alpha=alpha, noise_val=0.0)
            for b in betas_grid
        ],
        label=r'$S_{\tilde{\mathcal{E}}} (\theta)$',
        color=ct, 
        linewidth=lw, 
        alpha=ao, 
        linestyle='--'
    )

    ax[0].set_xlabel(r"$\theta$", fontsize=int(s * 1.2))
    ax[0].set_ylabel(r"Score Functions", fontsize=int(s * 1.2))
    ax[0].legend(prop={"size":s*0.9})

    disp_betas = [[], []]
    for i in range(len(betas[0])):
        vals = np.array([betas[j][i] for j in range(len(betas))])
        disp_betas[0].append(vals.min())
        disp_betas[1].append(vals.max())

    # We take the subset
    ax[1].fill_between(
        [i for i in range(len(disp_betas[0][int(5e4+1000):]))],
        disp_betas[0][int(5e4+1000):],
        disp_betas[1][int(5e4+1000):],
        alpha=0.3,
        color="orange",
    )

    ax[1].plot(means_betas[int(5e4+1000):], color=ct, linewidth=lw, label=r"$\langle\hat{\theta}\rangle$")
    ax[1].set_xscale("log")
    ax[1].axhline(beta_star, color="black", label=r"$\theta^*$")
    ax[1].set_ylabel(r"$\theta$", fontsize=int(s * 1.2))
    ax[1].set_xlabel(r"$t$", fontsize=int(s * 1.1))
    ax[1].legend(prop={"size":s*0.9})

    plt.tight_layout()

    plt.show()
    
    plt.plot(means_rts, label=r"$\mathcal{W}_d$", color=ca,linewidth=lw)
    disp_rts = [[], []]
    for i, rts_0i in enumerate(rts[0]):
        vals = np.array([rts_j[i] for j, rts_j in enumerate(rts)])
        disp_rts[0].append(vals.min())
        disp_rts[1].append(vals.max())

    plt.fill_between(
        list(range(len(means_rts))),
        disp_rts[0],
        disp_rts[1],
        alpha=0.5,
        color="orange",
    )

    #plt.axhline(1 - p_star, color="black", label="optimal reward")
    plt.ylabel("value", fontsize=int(s * 1.1))
    plt.xlabel(r"$t$", fontsize=int(s * 1.1))
    plt.legend(prop={"size":s})
    plt.show()

def set_and_run_experiment(q0s, rts, betas, alpha, lambd, noise_type, training_size):
    """Run a round of training from a previos stored experiment"""
    with open("data_rec/experiments/1/details.pickle", "rb") as f:
        details = pickle.load(f)

    betas_grid = details["betas"]

    mmin, p_star, beta_star = model_aware_optimal(
        betas_grid, alpha=alpha, lambd=lambd, noise_type=noise_type
    )

    details["alpha"] = [1.5, 0.25]  # No estoy seguro para que es esto.

    hyperparam = Hyperparameters(0.05, 0.99, 50, 150, 5000)
    # Run the full program and get the new dictionary with the changes.
    details = run_experiment(
        details,
        training_size,
        alpha,
        hyperparam,
        lambd=lambd,
        use_model=True,
        noise_type=noise_type,
    )
    q0s.append(details["tables"][0])
    betas.append(details["greed_beta"])
    rts.append(details["witness"])
    return betas_grid, beta_star, p_star


def __main__():
    """main"""
    cmd_args = read_cmd_args()
    # Set initial parameters
    amount_vals = cmd_args.amount_vals
    alpha = cmd_args.alpha
    lambd = cmd_args.lambd
    noise_type = cmd_args.noise_type
    training_size = cmd_args.training_size
    np.random.seed(seed=cmd_args.random_seed)  # Random seed

    betas = []
    rts = []
    q0s = []

    for i in range(amount_vals):
        betas_grid, beta_star, p_star = set_and_run_experiment(
            q0s, rts, betas, alpha, lambd, noise_type, training_size
        )

    make_plots(
        q0s,
        rts,
        betas,
        alpha,
        lambd,
        noise_type,
        betas_grid,
        beta_star,
        p_star,
    )


if __name__ == "__main__":
    __main__()

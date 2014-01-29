
import time
overall_start = time.time()

import nengo
import build
from nengo_ocl.sim_ocl import Simulator as SimOCL

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='Droid Serif')
import argparse
from mytools import hrr, nf, fh, nengo_stack_plot
import random

seed = 51001099090
random.seed(seed)

sim_class = nengo.Simulator
sim_class = SimOCL
run_time = 2.0

DperE = 32
dim = 256
num_ensembles = int(dim / DperE)
dim = num_ensembles * DperE

cleanup_n = 500
NperD = 30
NperE = NperD * DperE

max_rates=[200]
intercepts=[0.1]
radius=1.0


oja_scale = np.true_divide(20,1)
oja_learning_rate = np.true_divide(1,50)
pre_tau = 0.03
post_tau = 0.03


def run(sim_type, dim, DperE, cleanup_n):
    sim_class = {0: nengo.Simulator, 1: SimOCL}[sim_type]

    print "sim_class: ", sim_class, "dim: ", dim, "DperE: ", DperE, "cleanup_n: ", cleanup_n
    print "Building..."
    num_ensembles = int(dim / DperE)
    dim = num_ensembles * DperE
    NperE = NperD * DperE
    pre_max_rates=[400] * NperE
    pre_intercepts=[0.1] * NperE
    pre_radius=1.0
    ensemble_params={"radius":pre_radius,
                     "max_rates":pre_max_rates,
                     "intercepts":pre_intercepts}

    start = time.time()

    training_vector = np.array(hrr.HRR(dim).v)

    start = time.time()

# ----- Make Nodes -----
    model = nengo.Model("Oja Timing", seed=seed)
    with model:
        inn = nengo.Node(output=lambda x: [0] * dim)

        cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n), dimensions=dim,
                              max_rates=max_rates  * cleanup_n, intercepts=intercepts * cleanup_n,
                              radius=radius)

        pre_decoders = {'pre_' + str(i): np.zeros((dim, NperE)) for i in range(num_ensembles)}
        encoders = np.zeros((cleanup_n, dim))
        pre_ensembles, pre_decoders, pre_connections = \
                build.build_cleanup_oja(model, inn, cleanup, DperE, NperD, num_ensembles,
                                        ensemble_params, oja_learning_rate, oja_scale,
                                        pre_decoders=pre_decoders, encoders=encoders, pre_tau=pre_tau, post_tau=post_tau,
                                        end_time=None)

        cleanup_s = nengo.Probe(cleanup, 'spikes')
        #cleanup_w = nengo.Probe(pre_connections[0], 'transform')

        end = time.time()
        build_time = end - start
        print "Time:", build_time

# ----- Run and get data-----
    print "Running..."
    start = time.time()

    sim = sim_class(model, dt=0.001)
    sim.run(run_time)
    x = sim.data(cleanup_w)
    print x.shape

    end = time.time()
    sim_time = end - start
    print "Time:", sim_time

    return  sim_time, build_time



import time
overall_start = time.time()

import nengo
from nengo_ocl.sim_ocl import Simulator as SimOCL
import build

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='Droid Serif')

import argparse
from mytools import hrr, nf, fh, nengo_stack_plot
import random

sim_class = nengo.Simulator
sim_class = SimOCL
sim_length = 2

DperE = 2
dim = 16

cleanup_n = 20
NperD = 30

seed = 123
random.seed(seed)

learning_rate = np.true_divide(1,5)

def run(sim_type, dim, DperE, cleanup_n):
    sim_class = {0: nengo.Simulator, 1: SimOCL}[sim_type]
    input_vector = hrr.HRR(dim).v
    cleanup_encoders = np.array([input_vector] * cleanup_n)

    num_ensembles = int(dim / DperE)
    dim = num_ensembles * DperE
    NperE = NperD * DperE

    print "Building..."
    start = time.time()
    model = nengo.Model("PES timing", seed=seed)

    def input_func(x):
        return input_vector

    inn = nengo.Node(output=input_func)

# Build ensembles
    cleanup = nengo.Ensemble(label='cleanup', neurons = nengo.LIF(cleanup_n), dimensions=dim,
                            encoders=cleanup_encoders)

    nengo.Connection(inn, cleanup)

    ensembles = \
            build.build_cleanup_pes(cleanup, inn, DperE, NperD, num_ensembles, learning_rate)
    output_ensembles = ensembles[0]
    error_ensembles = ensembles[1]

# Build probes
    output_probes = [nengo.Probe(o, 'decoded_output', filter=0.1) for o in output_ensembles]
    error_probes = [nengo.Probe(e, 'decoded_output', filter=0.1) for e in error_ensembles]

    input_probe = nengo.Probe(inn, 'output')

    end = time.time()
    print "Build Time: ", end - start

# Run model
    print "Simulating..."
    start = time.time()
    sim = sim_class(model, dt=0.001)
    sim.run(sim_length)
    end = time.time()
    print "Sim Time: ", end - start

    end = time.time()
    print "Plot Time: ", end - start

    overall_end = time.time()
    print "Total time: ", overall_end - overall_start

    plt.show()



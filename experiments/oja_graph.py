import time
overall_start = time.time()

import nengo
import build


import numpy as np
import matplotlib.pyplot as plt
plt.copper()
import argparse
from mytools import hrr, nf, fh, nengo_plot_helper, extract_probe_data
import random

from matplotlib import rc, font_manager
plt.rc('text', usetex=True)
plt.rc('axes', color_cycle=['gray'])

seed = 510
random.seed(seed)

sim_class = nengo.Simulator
learning_time = 2 #in seconds
testing_time = 0.1 #in seconds
ttms = testing_time * 1000 #in ms
hrr_num = 1

DperE = 64
dim = 64
num_ensembles = int(dim / DperE)
dim = num_ensembles * DperE

cleanup_n = 1
NperD = 30
NperE = NperD * DperE

max_rates=[200]
intercepts=[0.2]
radius=1.0

pre_max_rates=[400] * NperE
pre_intercepts=[0.1] * NperE
pre_radius=1.0
ensemble_params={"radius":pre_radius,
                 "max_rates":pre_max_rates,
                 "intercepts":pre_intercepts}

oja_scale = np.true_divide(7,1)
oja_learning_rate = np.true_divide(1,50)
pre_tau = 0.03
post_tau = 0.03

file_config = {
                'seed':seed,
                'NperD':NperD,
                'dim':dim,
                'DperE': DperE,
                'int':intercepts[0],
                'maxrates':max_rates[0],
                'radius':radius,
                'ojascale':oja_scale,
                'lr':oja_learning_rate,
                'hrrnum':hrr_num,
                'learntime':learning_time,
                'testtime':testing_time,
              }

filename = fh.make_filename("oja_graph_data", directory="oja_graph_data",
                            use_time=False, config_dict=file_config, extension=".npz")

run_sim = False

try:
    print "Trying to load..."
    f = open(filename, 'r')
    with np.load(f) as npz:
        sims, sims_time = npz['sims'], npz['sims_time']
        before_spikes, before_time = npz['before_spikes'], npz['before_time']
        after_spikes, after_time = npz['after_spikes'], npz['after_time']
    print "Loaded"
except:
    print "Couldn't load."
    run_sim = True

if run_sim:

    print "Building..."
    start = time.time()
    model = nengo.Model("Network Array OJA", seed=seed)

    training_vector = np.array(hrr.HRR(dim).v)
    ortho = nf.ortho_vector(training_vector)
    ortho2 = nf.ortho_vector(training_vector)

    hrr_noise = nf.make_hrr_noise(dim, hrr_num)
    noisy_vector = hrr_noise(training_vector)
    print "HRR sim: ", np.dot(noisy_vector, training_vector)

# --- Build input for phase 1
    gens1 = [
            nf.interpolator(1, ortho, training_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, training_vector,
               ortho, lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, ortho, noisy_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, noisy_vector,
               ortho, lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, ortho2, training_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, training_vector,
               ortho2, lambda x: np.true_divide(x, ttms)),
            ]
    times1 = [testing_time] * 6
    phase1_input = nf.make_f(gens1, times1)

# --- Build input for phase 2
    gens2 = [
            nf.output(100, True, training_vector, False),
            nf.output(100, True, np.zeros(dim), False),
            nf.interpolator(1, ortho, training_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, training_vector,
               ortho, lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, ortho, noisy_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, noisy_vector,
               ortho, lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, ortho2, training_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, training_vector,
               ortho2, lambda x: np.true_divide(x, ttms)),
            ]

    times2 = [0.9 * learning_time , 0.1 * learning_time] + [testing_time] * 6
    phase2_input = nf.make_f(gens2, times2)

    orthos = [ortho]
    if cleanup_n > 1:
        orthos.append(ortho2)
    if cleanup_n > 2:
        orthos.extend([nf.ortho_vector(training_vector) for i in range(max(cleanup_n - 2, 0))])

    p = 0.3
    encoders = [p * training_vector + (1-p) * o for o in orthos]
    encoders = np.array([enc / np.linalg.norm(enc) for enc in encoders])
    print [np.dot(enc,training_vector) for enc in encoders]

# PHASE 1 - ******************************************
    print "Building phase 1"
    start = time.time()

# ----- Make Nodes -----
    model = nengo.Model("Phase 1", seed=seed)

    inn = nengo.Node(output=phase1_input)

    cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n), dimensions=dim,
                          max_rates=max_rates  * cleanup_n, intercepts=intercepts * cleanup_n,
                          encoders=encoders, radius=radius)

    pre_ensembles, pre_decoders, pre_connections = \
            build.build_cleanup_oja(model, inn, cleanup, DperE, NperD, num_ensembles,
                                    ensemble_params, oja_learning_rate, oja_scale,
                                    use_oja=False)
    cleanup1_s = nengo.Probe(cleanup, 'spikes')
    inn_p = nengo.Probe(inn, 'output')

    end = time.time()
    print "Time:", end - start

# ----- Run and get data-----
    print "Running phase 1"
    start = time.time()

    sim1 = sim_class(model, dt=0.001)
    sim1.run(sum(times1))

    end = time.time()
    print "Time:", end - start


# PHASE 2 - *******************************************
    print "Building phase 2"
    start = time.time()

# ----- Make Nodes -----
    model = nengo.Model("Phase 2", seed=seed)
    with model:
        inn = nengo.Node(output=phase2_input)

        cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n), dimensions=dim,
                              max_rates=max_rates  * cleanup_n, intercepts=intercepts * cleanup_n,
                              encoders=encoders, radius=radius)

        pre_ensembles, pre_decoders, pre_connections = \
                build.build_cleanup_oja(model, inn, cleanup, DperE, NperD, num_ensembles,
                                        ensemble_params, oja_learning_rate, oja_scale,
                                        pre_decoders=pre_decoders, pre_tau=pre_tau, post_tau=post_tau,
                                        end_time=learning_time,)
                                        #use_oja=False)

        cleanup2_s = nengo.Probe(cleanup, 'spikes')

        if 0:
            weight_probes = [nengo.Probe(pc, 'transform') for pc in pre_connections]

        end = time.time()
        print "Time:", end - start

# ----- Run and get data-----
    print "Running phase 2"
    start = time.time()

    sim2 = sim_class(model, dt=0.001)
    sim2.run(sum(times2))

    end = time.time()
    print "Time:", end - start

    t1 = sim1.trange()
    t2 = sim2.trange()

    sim_func = lambda x: np.dot(x, training_vector)
    sims, sims_time = extract_probe_data(t1, sim1, inn_p, func=sim_func)

    before_spikes, before_time = extract_probe_data(t1, sim1, cleanup1_s, spikes=True)

    test_slice = np.index_exp[-len(t1):][0]
    after_spikes, after_time = extract_probe_data(t2, sim2, cleanup2_s, slice=test_slice, spikes=True)

    f = open(filename, 'w')
    np.savez(f, sims=sims, sims_time=sims_time,
                before_spikes=before_spikes, before_time=before_time,
                after_spikes=after_spikes, after_time=after_time)

# ----- Plot! -----
print "Plotting..."
start = time.time()

num_plots = 3

offset = num_plots * 100 + 10 + 1

xticks = [2 * i * testing_time for i in range(4)]

ax, offset = nengo_plot_helper(offset, sims_time, sims, \
                               label=r'$Similarity$', yticks=[0,1])
plt.xticks(xticks)
ax.xaxis.set_ticklabels([])

ax, offset = nengo_plot_helper(offset, before_time, before_spikes, yticks=[], spikes=True)
plt.xticks(xticks)
ax.xaxis.set_ticklabels([])
plt.ylabel(r'$Before\ Training$')

ax, offset = nengo_plot_helper(offset, after_time, after_spikes, yticks=[], spikes=True)
plt.xlabel(r'$Time\ (s)$')
plt.ylabel(r'$After\ Training$')
plt.xticks(xticks)

file_config = {
                'seed':seed,
                'NperD':NperD,
                'dim':dim,
                'DperE': DperE,
                'int':intercepts[0],
                'maxrates':max_rates[0],
                'radius':radius,
                'ojascale':oja_scale,
                'lr':oja_learning_rate,
                'hrrnum':hrr_num,
                'learntime':learning_time,
                'testtime':testing_time,
              }

filename = fh.make_filename('oja_graphs', directory='oja_graphs',
                            config_dict=file_config, extension='.pdf')
plt.savefig(filename)

end = time.time()
print "Time:", end - start

overall_end = time.time()
print "Total time: ", overall_end - overall_start

plt.show()


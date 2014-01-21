import time
overall_start = time.time()

import nengo
import build

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='Droid Serif')

import argparse
from mytools import hrr, nf, fh, nengo_stack_plot
import random

sim_class = nengo.Simulator
sim_length = 2
learning_time = 2 #in seconds
testing_time = 0.25 #in seconds
ttms = testing_time * 1000 #in ms
hrr_num = 1

DperE = 2
dim = 8
num_ensembles = int(dim / DperE)
dim = num_ensembles * DperE

cleanup_n = 1
NperD = 30
NperE = NperD * DperE

seed = 123
random.seed(seed)

oja_scale = np.true_divide(10,1)
oja_learning_rate = np.true_divide(1,50)

input_vector = hrr.HRR(dim).v

print "Building..."
start = time.time()
model = nengo.Model("Network Array PES", seed=seed)

max_rates=[200]
intercepts=[0.15]

pre_max_rates=[200]
pre_radius=0.5
pre_intercepts=[0.10]
ensemble_params={"radius":pre_radius,
                 "max_rates":pre_max_rates,
                 "intercepts":pre_intercepts}

training_vector = np.array(hrr.HRR(dim).v)
ortho = nf.ortho_vector(training_vector)
ortho2 = nf.ortho_vector(training_vector)

hrr_noise = nf.make_hrr_noise(dim, hrr_num)
noisy_vector = hrr_noise(training_vector)

# --- Build input for phase 1
gens1 = [
        nf.interpolator(1, training_vector,
           ortho, lambda x: np.true_divide(x, ttms)),
        nf.interpolator(1, ortho, training_vector,
           lambda x: np.true_divide(x, ttms)),
        nf.interpolator(1, noisy_vector,
           ortho, lambda x: np.true_divide(x, ttms)),
        nf.interpolator(1, ortho, noisy_vector,
           lambda x: np.true_divide(x, ttms)),
        nf.interpolator(1, training_vector,
           ortho2, lambda x: np.true_divide(x, ttms)),
        nf.interpolator(1, ortho2, training_vector,
           lambda x: np.true_divide(x, ttms)),
        ]

times1 = [testing_time] * 6
phase1_input = nf.make_f(gens1, times1)

# --- Build input for phase 2
gens2 = [
        nf.output(100, True, training_vector, False),
        nf.interpolator(1, training_vector,
           ortho, lambda x: np.true_divide(x, ttms)),
        nf.interpolator(1, ortho, training_vector,
           lambda x: np.true_divide(x, ttms)),
        nf.interpolator(1, noisy_vector,
           ortho, lambda x: np.true_divide(x, ttms)),
        nf.interpolator(1, ortho, noisy_vector,
           lambda x: np.true_divide(x, ttms)),
        nf.interpolator(1, training_vector,
           ortho2, lambda x: np.true_divide(x, ttms)),
        nf.interpolator(1, ortho2, training_vector,
           lambda x: np.true_divide(x, ttms))
        ]

times2 = [learning_time] + [testing_time] * 6
phase2_input = nf.make_f(gens2, times2)

#make the neuron start with an encoder different from the vector
#we train on, since thats what will happen in the real model
p = 0.25
encoders = np.array([p * training_vector + (1-p) * ortho])
encoders[0] = encoders[0] / np.linalg.norm(encoders[0])
print np.dot(encoders,training_vector)

# PHASE 1 - ******************************************
print "Building phase 1"
start = time.time()

# ----- Make Nodes -----
model = nengo.Model("Phase 1", seed=seed)

inn = nengo.Node(output=phase1_input)

cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n), dimensions=dim,
                      max_rates=max_rates  * cleanup_n, intercepts=intercepts * cleanup_n, encoders=encoders)

pre_ensembles, pre_decoders = \
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

sim1 = nengo.Simulator(model, dt=0.001)
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
                          max_rates=max_rates  * cleanup_n, intercepts=intercepts * cleanup_n, encoders=encoders)

    pre_ensembles, pre_decoders = \
            build.build_cleanup_oja(model, inn, cleanup, DperE, NperD, num_ensembles,
                                    ensemble_params, oja_learning_rate, oja_scale,
                                    pre_decoders=pre_decoders)

    cleanup2_s = nengo.Probe(cleanup, 'spikes')

    #oja_pre_p = nengo.Probe(oja_rule, 'pre')
    #oja_post_p = nengo.Probe(oja_rule, 'post')
    #oja_oja_p = nengo.Probe(oja_rule, 'oja')
    #oja_delta_p = nengo.Probe(oja_rule, 'delta')
    #weights_p = nengo.Probe(conn, 'transform')

    end = time.time()
    print "Time:", end - start

# ----- Run and get data-----
print "Running phase 2"
start = time.time()

sim2 = nengo.Simulator(model, dt=0.001)
sim2.run(sum(times2))

end = time.time()
print "Time:", end - start

# ----- Plot! -----
print "Plotting..."
start = time.time()

if 0:
    num_plots = 9
else:
    num_plots = 5

offset = num_plots * 100 + 10 + 1

t1 = sim1.trange()
t2 = sim2.trange()

sim_func = lambda x: np.dot(x, training_vector)
ax, offset = nengo_stack_plot(offset, t1, sim1, inn_p, func=sim_func, label='Similarity')
ax, offset = nengo_stack_plot(offset, t1, sim1, inn_p, label='Input')

ax, offset = nengo_stack_plot(offset, t1, sim1, cleanup1_s, label='Spikes: Testing 1')

test_slice = np.index_exp[-len(t1):][0]
ax, offset = nengo_stack_plot(offset, t2, sim2, cleanup2_s,
                              label='Spikes: Testing 2', slice=test_slice)

learn_slice = np.index_exp[:-len(t1)][0]
ax, offset = nengo_stack_plot(offset, t2, sim2, cleanup2_s,
                              label='Spikes: Learning', slice=learn_slice)
#
#if 0:
#    plt.subplot(offset)
#    plt.plot(t2[0:-len(t1)], sim.data(oja_pre_p)[0:-len(t1)])
#    plt.ylabel('Oja pre')
#    offset += 1
#
#    plt.subplot(offset)
#    plt.plot(t2[0:-len(t1)], sim.data(oja_post_p)[0:-len(t1)])
#    plt.ylabel('Oja post')
#    offset += 1
#
#    plt.subplot(offset)
#    plt.plot(t2[0:-len(t1)], np.squeeze(sim.data(oja_delta_p)[0:-len(t1)]))
#    plt.ylabel('Oja delta')
#    offset += 1
#
#    plt.subplot(offset)
#    plt.plot(t2[0:-len(t1)], np.squeeze(sim.data(oja_oja_p)[0:-len(t1)]))
#    plt.ylabel('Oja oja')
#    offset += 1
#
#    plt.subplot(offset)
#    plt.plot(t2[0:-len(t1)], np.squeeze(sim.data(weights_p)[0:-len(t1)]))
#    plt.ylabel('Oja weights')
#    offset += 1

file_config = {
                'NperE':NperE,
                'numEnsembles':num_ensembles,
                'dim':dim,
                'DperE': DperE,
                'cleanupN': cleanup_n,
                'premaxr':max_rates[0],
                'preint':pre_intercepts[0],
                'int':intercepts[0],
                'ojascale':oja_scale,
                'lr':oja_learning_rate,
                'hrrnum':hrr_num,
              }

filename = fh.make_filename('oja_network_array', directory='oja_network_array',
                            config_dict=file_config, extension='.png')
plt.savefig(filename)

end = time.time()
print "Time:", end - start

overall_end = time.time()
print "Total time: ", overall_end - overall_start

plt.show()


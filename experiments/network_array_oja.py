
import time
overall_start = time.time()

# The purpose  of this script is to get OJA selectivity enhancement working
# when the input is a network array
import nengo
from nengo.matplotlib import rasterplot
from nengo.nonlinearities import OJA, PES
from nengo_ocl.sim_ocl import Simulator as SimOCL
from nengo_ocl.sim_npy import Simulator as SimNumpy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='Droid Serif')

import argparse
from mytools import hrr, nf, fh
import random
import pprint

seed = 8101000
sim_class = nengo.Simulator

max_rates=[200]
intercepts=[0.15]
pre_max_rates=[200]
pre_radius=0.5
pre_intercepts=[0.10]

oja_scale = np.true_divide(10,1)
oja_learning_rate = np.true_divide(1,50)

learning_time = 2 #in seconds
testing_time = 0.25 #in seconds
ttms = testing_time * 1000 #in ms
hrr_num = 1

dim_per_ensemble = 32
dim = 64
num_ensembles = int(dim / dim_per_ensemble)
dim = num_ensembles * dim_per_ensemble

cleanup_n = 1
ensemble_nperd = 30
npere = ensemble_nperd * dim_per_ensemble
total_n = npere * num_ensembles

random.seed(seed)

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
encoder = np.array([p * training_vector + (1-p) * ortho])
encoder[0] = encoder[0] / np.linalg.norm(encoder[0])
print np.dot(encoder,training_vector)

# PHASE 1 - ******************************************
print "Building phase 1"
start = time.time()

# ----- Make Nodes -----
model = nengo.Model("Phase 1", seed=seed)

# -- Get Decoders
pre_ensembles = []
for i in range(num_ensembles):
    pre_ensembles.append(nengo.Ensemble(label='pre_'+str(i), neurons=nengo.LIF(npere),
                        dimensions=dim_per_ensemble,
                        intercepts=pre_intercepts * npere,
                        max_rates=pre_max_rates * npere,
                        radius=pre_radius))
dummy = nengo.Ensemble(label='dummy', neurons=nengo.LIF(npere),
                        dimensions=dim)
def make_func(dim, start):
    def f(x):
        y = np.zeros(dim)
        y[start:start+len(x)] = x
        return y
    return f

for i,pre in enumerate(pre_ensembles):
    nengo.Connection(pre, dummy, function=make_func(dim, i * dim_per_ensemble))

sim = nengo.Simulator(model, dt=0.001)
sim.run(.01)

pre_decoders = {}
for conn in sim.model.connections:
    if conn.pre.label.startswith('pre'):
        pre_decoders[conn.pre.label] = conn._decoders

# -- Build the rest of the nodes
inn = nengo.Node(output=phase1_input)
cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n), dimensions=dim,
                      max_rates=max_rates  * cleanup_n, intercepts=intercepts * cleanup_n, encoders=encoder)

# ----- Make Connections -----
in_transform=np.eye(dim_per_ensemble)
in_transform = np.concatenate((in_transform, np.zeros((dim_per_ensemble, dim - dim_per_ensemble))), axis=1)
for pre in pre_ensembles:
    nengo.Connection(inn, pre, transform=in_transform)
    in_transform = np.roll(in_transform, dim_per_ensemble, axis=1)

    connection_weights = np.dot(encoder, pre_decoders[pre.label])
    conn = nengo.Connection(pre.neurons, cleanup.neurons, transform=connection_weights)

# ----- Make Probe -----
pre_probes = []
for pre in pre_ensembles:
    pre_probes.append(nengo.Probe(pre, 'decoded_output', filter=0.05))

cleanup_s = nengo.Probe(cleanup, 'spikes')
inn_p = nengo.Probe(inn, 'output')

end = time.time()
print "Time:", end - start

# ----- Run and get data-----
print "Running phase 1"
start = time.time()

sim = nengo.Simulator(model, dt=0.001)
sim.run(sum(times1))
t1 = sim.trange()
spikes1 = sim.data(cleanup_s)
inn1 = sim.data(inn_p)
pre1 = np.zeros((len(t1), 0))
for pre_p in pre_probes:
    pre1 = np.concatenate((pre1, sim.data(pre_p)), axis=1)

end = time.time()
print "Time:", end - start


# PHASE 2 - *******************************************
print "Building phase 2"
start = time.time()

# ----- Make Nodes -----
model = nengo.Model("Phase 2", seed=seed)
with model:
    inn = nengo.Node(output=phase2_input)

    pre_ensembles = []
    for i in range(num_ensembles):
        pre_ensembles.append(nengo.Ensemble(label='pre_'+str(i), neurons=nengo.LIF(npere),
                            dimensions=dim_per_ensemble,
                            intercepts=pre_intercepts * npere,
                            max_rates=pre_max_rates * npere,
                            radius=pre_radius))

    cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n),
                          dimensions=dim,
                          max_rates=max_rates  * cleanup_n,
                          intercepts=intercepts * cleanup_n,
                          encoders=encoder)

# ----- Make Connections -----
    in_transform=np.eye(dim_per_ensemble)
    in_transform = np.concatenate((in_transform, np.zeros((dim_per_ensemble, dim - dim_per_ensemble))), axis=1)
    for pre in pre_ensembles:

        nengo.Connection(inn, pre, transform=in_transform)
        in_transform = np.roll(in_transform, dim_per_ensemble, axis=1)

        oja_rule = OJA(pre_tau=0.05, post_tau=0.05,
                learning_rate=oja_learning_rate, oja_scale=oja_scale)
        connection_weights = np.dot(encoder, pre_decoders[pre.label])
        conn = nengo.Connection(pre.neurons, cleanup.neurons,
                                transform=connection_weights, learning_rule=oja_rule)

# ----- Make Probe -----
    pre_probes = []
    for pre in pre_ensembles:
        pre_probes.append(nengo.Probe(pre, 'decoded_output', filter=0.05))

    cleanup_s = nengo.Probe(cleanup, 'spikes')
    inn_p = nengo.Probe(inn, 'output')
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

sim = nengo.Simulator(model, dt=0.001)
sim.run(sum(times2))
t2 = sim.trange()
spikes2 = sim.data(cleanup_s)
inn2 = sim.data(inn_p)
pre2 = np.zeros((len(t2), 0))
for pre_p in pre_probes:
    pre2 = np.concatenate((pre2, sim.data(pre_p)), axis=1)

end = time.time()
print "Time:", end - start

# ----- Plot! -----
print "Plotting..."
start = time.time()

plot_neuron_limit = 30

#if total_n <= plot_neuron_limit:
if 0:
    num_plots = 9
else:
    num_plots = 7
offset = num_plots * 100 + 10 + 1

ax = plt.subplot(offset)
sims = np.array([np.dot(i, training_vector) for i in inn2[-len(t1):]])
plt.plot(t1, sims)
plt.xlim((min(t1), max(t1)))
plt.ylabel('Similarity to training_vector')
plt.title('Clean then noisy')
offset += 1

plt.subplot(offset)
plt.plot(t1, inn1)
plt.xlim((min(t1), max(t1)))
plt.ylabel('input')
offset += 1

#Testing Phase 1
plt.subplot(offset)
rasterplot(t1, spikes1)
plt.ylabel('Phase 1: Spikes')
offset += 1

#Testing Phase 2
plt.subplot(offset)
rasterplot(t1, spikes2[-len(t1):])
plt.ylabel('Phase 2: Spikes')
offset += 1

#Learning
plt.subplot(offset)
rasterplot(t2[0:-len(t1)], spikes2[0:-len(t1)])
plt.ylabel('Learning: Spikes')
offset += 1

#plt.subplot(offset)
#plt.plot(t1, pre1)
#plt.ylabel('Pre Output')
#offset += 1
#
#plt.subplot(offset)
#plt.plot(t2, pre2)
#plt.ylabel('Pre Output')
#offset += 1




#if total_n <= plot_neuron_limit:
if 0:
    plt.subplot(offset)
    plt.plot(t2[0:-len(t1)], sim.data(oja_pre_p)[0:-len(t1)])
    plt.ylabel('Oja pre')
    offset += 1

    plt.subplot(offset)
    plt.plot(t2[0:-len(t1)], sim.data(oja_post_p)[0:-len(t1)])
    plt.ylabel('Oja post')
    offset += 1

    plt.subplot(offset)
    plt.plot(t2[0:-len(t1)], np.squeeze(sim.data(oja_delta_p)[0:-len(t1)]))
    plt.ylabel('Oja delta')
    offset += 1

    plt.subplot(offset)
    plt.plot(t2[0:-len(t1)], np.squeeze(sim.data(oja_oja_p)[0:-len(t1)]))
    plt.ylabel('Oja oja')
    offset += 1

    plt.subplot(offset)
    plt.plot(t2[0:-len(t1)], np.squeeze(sim.data(weights_p)[0:-len(t1)]))
    plt.ylabel('Oja weights')
    offset += 1

file_config = {
                'NperE':npere,
                'numEnsembles':num_ensembles,
                'dim':dim,
                'DperE': dim_per_ensemble,
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


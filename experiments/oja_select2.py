

# The purpose  of this script is to show that the OJA learning rule does indeed
# increase the selectivity of the neuron. Good for getting the parameters on
# OJA correct.
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

seed = 810
dim = 128
max_rates=[200]
intercepts=[0.25]

ensemble_n = 30 * dim
pre_max_rates=[200]
pre_radius=0.5
pre_intercepts=[0.10]

oja_learning_rate = np.true_divide(1,50)

learning_time = 3 #in seconds
testing_time = 0.25 #in seconds
ttms = testing_time * 1000 #in ms
oja_scale = np.true_divide(10,1)
hrr_num = 1

random.seed(seed)

training_vector = np.array(hrr.HRR(dim).v)
ortho = nf.ortho_vector(training_vector)
ortho2 = nf.ortho_vector(training_vector)

hrr_noise = nf.make_hrr_noise(dim, hrr_num)
noisy_vector = hrr_noise(training_vector)

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

#make the neuron start with an encoder different from the vector
#we train on, since thats what will happen in the real model
p = 0.25
encoder = np.array([p * training_vector + (1-p) * ortho])
encoder[0] = encoder[0] / np.linalg.norm(encoder[0])
print np.dot(encoder,training_vector)

times2 = [learning_time] + [testing_time] * 6
phase2_input = nf.make_f(gens2, times2)

# PHASE 1 - ******************************************
print "Building phase 1"

# ----- Make Nodes -----
model = nengo.Model("Phase 1", seed=seed)

# -- Get Decoders
pre = nengo.Ensemble(label='pre', neurons=nengo.LIF(ensemble_n), dimensions=dim,
                     intercepts=pre_intercepts * ensemble_n,
                     max_rates=pre_max_rates * ensemble_n,
                     radius=pre_radius)
pre_p = nengo.Probe(pre, 'decoded_output')
sim = nengo.Simulator(model, dt=0.001)
sim.run(.01)
pre_decoders = sim.model.connections[0]._decoders

# -- Build the rest of the nodes
inn = nengo.Node(output=phase1_input)
cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(1), dimensions=dim,
                      max_rates=max_rates  * 1, intercepts=intercepts * 1, encoders=encoder)

# ----- Make Connections -----
nengo.Connection(inn, pre)

transform = np.dot(encoder, pre_decoders)
conn = nengo.Connection(pre.neurons, cleanup.neurons, transform=transform)

# ----- Make Probe -----
cleanup_s = nengo.Probe(cleanup, 'spikes')
inn_p = nengo.Probe(inn, 'output')

# ----- Run and get data-----
print "Running phase 1"
sim = nengo.Simulator(model, dt=0.001)
sim.run(sum(times1))
t1 = sim.trange()
spikes1 = sim.data(cleanup_s)
inn1 = sim.data(inn_p)


# PHASE 2 - *******************************************
print "Building phase 2"

#pre_decoders = sim.model.connections[1]._decoders
gain = filter(lambda x: getattr(x,'label',0) == 'cleanup', sim.model.objs)
gain = gain[0].neurons.gain
#sums = np.sum(gain * np.abs(transform)/pre_radius, 1)
sums = np.sum(gain * transform * transform/pre_radius, 1)
print "max sums:", max(sums)
print "min sums:", min(sums)
print "mean sums:", np.mean(sums)

# ----- Make Nodes -----
model = nengo.Model("Phase 2", seed=seed)
with model:
    inn = nengo.Node(output=phase2_input)

    pre = nengo.Ensemble(label='pre', neurons=nengo.LIF(ensemble_n), dimensions=dim,
                     intercepts=pre_intercepts * ensemble_n,
                     max_rates=pre_max_rates * ensemble_n,
                     radius=pre_radius)
    cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(1),
                          dimensions=dim,
                          max_rates=max_rates  * 1,
                          intercepts=intercepts * 1, encoders=encoder)

# ----- Make Connections -----
    nengo.Connection(inn, pre)

    oja_rule = OJA(pre_tau=0.05, post_tau=0.05,
            learning_rate=oja_learning_rate, oja_scale=oja_scale)
    transform = np.dot(encoder, pre_decoders)
    conn = nengo.Connection(pre.neurons, cleanup.neurons,
                            transform=transform, learning_rule=oja_rule)

# ----- Make Probe -----
    cleanup_s = nengo.Probe(cleanup, 'spikes')
    inn_p = nengo.Probe(inn, 'output')
    oja_pre_p = nengo.Probe(oja_rule, 'pre')
    oja_post_p = nengo.Probe(oja_rule, 'post')
    oja_oja_p = nengo.Probe(oja_rule, 'oja')
    oja_delta_p = nengo.Probe(oja_rule, 'delta')
    weights_p = nengo.Probe(conn, 'transform')

# ----- Run and get data-----
print "Running phase 2"
sim = nengo.Simulator(model, dt=0.001)
sim.run(sum(times2))
t2 = sim.trange()
spikes2 = sim.data(cleanup_s)
inn2 = sim.data(inn_p)


# ----- Plot! -----
print "Plotting..."

if ensemble_n <= 300:
    num_plots = 9
else:
    num_plots = 4
offset = num_plots * 100 + 10 + 1

plt.subplot(offset)
plt.plot(t1, [np.dot(i, training_vector) for i in inn2[-len(t1):]])
plt.ylabel('Similarity to training_vector')
plt.title('Clean then noisy')
offset += 1

#Testing Phase 1
plt.subplot(offset)
rasterplot(t1, spikes1)
plt.ylabel('Cleanup: Spikes')
offset += 1

#Testing Phase 2
plt.subplot(offset)
rasterplot(t1, spikes2[-len(t1):])
plt.ylabel('Cleanup: Spikes')
offset += 1

#Learning
plt.subplot(offset)
rasterplot(t2[0:-len(t1)], spikes2[0:-len(t1)])
plt.ylabel('Cleanup: Spikes')
offset += 1

if ensemble_n <= 300:
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
                'ensembleN':ensemble_n,
                'premaxr':max_rates[0],
                'preint':pre_intercepts[0],
                'int':intercepts[0],
                'dim':dim,
                'ojascale':oja_scale,
                'lr':oja_learning_rate,
                'hrrnum':hrr_num,
              }

filename = fh.make_filename('oja_select', directory='oja_select',
                            config_dict=file_config, extension='.png')
plt.savefig(filename)

plt.show()

import time
overall_start = time.time()

import nengo
import build
from nengo.helpers import tuning_curves

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
learning_time = 2 #in seconds
testing_time = 0.25 #in seconds
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
intercepts=[0.1]
radius=1.0

pre_max_rates=[400] * NperE
pre_intercepts=[0.1] * NperE
pre_radius=1.0
ensemble_params={"radius":pre_radius,
                 "max_rates":pre_max_rates,
                 "intercepts":pre_intercepts}

oja_scale = np.true_divide(20,1)
oja_learning_rate = np.true_divide(1,50)
pre_tau = 0.03
post_tau = 0.03

print "Building..."
start = time.time()
model = nengo.Model("Network Array PES", seed=seed)

training_vector = np.array(hrr.HRR(dim).v)
ortho = nf.ortho_vector(training_vector)
ortho2 = nf.ortho_vector(training_vector)

hrr_noise = nf.make_hrr_noise(dim, hrr_num)
noisy_vector = hrr_noise(training_vector)
print "HRR sim: ", np.dot(noisy_vector, training_vector)

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
p = 0.3
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

sim1 = nengo.Simulator(model, dt=0.001)
sim1.run(sum(times1))

#eval_points, activities = tuning_curves(next(o for o in sim1.model.objs if o.label=='cleanup'))
#print eval_points.shape
#print activities.shape
#plt.plot([np.dot(e, encoders[0]) for e in eval_points], activities)
#plt.show()

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
                                    end_time=learning_time)

    cleanup2_s = nengo.Probe(cleanup, 'spikes')

    #oja_pre_p = nengo.Probe(oja_rule, 'pre')
    #oja_post_p = nengo.Probe(oja_rule, 'post')
    #oja_oja_p = nengo.Probe(oja_rule, 'oja')
    #oja_delta_p = nengo.Probe(oja_rule, 'delta')
    weight_probes = [nengo.Probe(pc, 'transform') for pc in pre_connections]

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


hrr_noise = nf.make_hrr_noise(dim, 2)
noisy_vector2 = hrr_noise(training_vector)

hrr_noise = nf.make_hrr_noise(dim, 3)
noisy_vector3 = hrr_noise(training_vector)
num_plots = 4
offset = num_plots*100 + 11

connection_weights_start = [np.squeeze(sim2.data(wp)[0,:,:], axis=(0,)) for wp in weight_probes]
connection_weights_start = reduce(lambda x,y: np.concatenate((x,y), axis=1), connection_weights_start)
print "sum of squares: start"
print sum (connection_weights_start**2)

connection_weights = [np.squeeze(sim2.data(wp)[int((learning_time - 0.2) * 1000),:,:], axis=(0,)) for wp in weight_probes]
connection_weights = reduce(lambda x,y: np.concatenate((x,y), axis=1), connection_weights)
print "sum of squares: end"
print sum (connection_weights**2)

pre_encoders = next(ens.encoders for ens in sim2.model.objs if ens.label.startswith('pre'))
plt.subplot(offset)
dots = np.dot(pre_encoders, training_vector[:,np.newaxis])
plt.scatter(dots, connection_weights_start, color='red', alpha=0.5, label='start')
plt.scatter(dots, connection_weights, color='blue', alpha=0.5, label='end')
plt.title("Clean")
plt.xlim((-1.0, 1.0))
plt.ylim((-1.0, 1.0))
plt.legend()
offset += 1
plt.subplot(offset)
dots = np.dot(pre_encoders, noisy_vector[:,np.newaxis])
plt.scatter(dots, connection_weights_start, color='red', alpha=0.5, label='start')
plt.scatter(dots, connection_weights, color='blue', alpha=0.5, label='end')
plt.title("Noisy")
plt.xlim((-1.0, 1.0))
plt.ylim((-1.0, 1.0))
plt.legend()
offset += 1
#plt.subplot(offset)
#dots = np.dot(pre_encoders, noisy_vector2[:,np.newaxis])
#plt.scatter(dots, connection_weights_start, color='red', alpha=0.5, label='start')
#plt.scatter(dots, connection_weights, color='blue', alpha=0.5, label='end')
#plt.title("Noisy")
#plt.xlim((-1.0, 1.0))
#plt.ylim((-1.0, 1.0))
#plt.legend()
#offset += 1
#plt.subplot(offset)
#dots = np.dot(pre_encoders, noisy_vector3[:,np.newaxis])
#plt.scatter(dots, connection_weights_start, color='red', alpha=0.5, label='start')
#plt.scatter(dots, connection_weights, color='blue', alpha=0.5, label='end')
#plt.title("Noisy")
#plt.xlim((-1.0, 1.0))
#plt.ylim((-1.0, 1.0))
#plt.legend()
#offset += 1
plt.subplot(offset)
dots = np.dot(pre_encoders, ortho[:,np.newaxis])
plt.scatter(dots, connection_weights_start, color='red', alpha=0.5, label='start')
plt.scatter(dots, connection_weights, color='blue', alpha=0.5, label='end')
plt.title("ortho1")
plt.xlim((-1.0, 1.0))
plt.ylim((-1.0, 1.0))
plt.legend()
offset += 1
plt.subplot(offset)
dots = np.dot(pre_encoders, ortho2[:,np.newaxis])
plt.scatter(dots, connection_weights_start, color='red', alpha=0.5, label='start')
plt.scatter(dots, connection_weights, color='blue', alpha=0.5, label='end')
plt.title("ortho2")
plt.xlim((-1.0, 1.0))
plt.ylim((-1.0, 1.0))
plt.legend()
plt.show()

num_plots = 6

offset = num_plots * 100 + 10 + 1

t1 = sim1.trange()
t2 = sim2.trange()

sim_func = lambda x: np.dot(x, training_vector)
ax, offset = nengo_stack_plot(offset, t1, sim1, inn_p, func=sim_func, label='Similarity to training')
sim_func = lambda x: np.dot(x, encoders[0])
ax, offset = nengo_stack_plot(offset, t1, sim1, inn_p, func=sim_func, label='Similarity to encoder')

ax, offset = nengo_stack_plot(offset, t1, sim1, cleanup1_s, label='Spikes: Testing 1')

test_slice = np.index_exp[-len(t1):][0]
ax, offset = nengo_stack_plot(offset, t2, sim2, cleanup2_s,
                              label='Spikes: Testing 2', slice=test_slice)

learn_slice = np.index_exp[:-len(t1)][0]
ax, offset = nengo_stack_plot(offset, t2, sim2, cleanup2_s,
                              label='Spikes: Learning', slice=learn_slice)

ax, offset = nengo_stack_plot(offset, t2, sim2, weight_probes,
                              label='Connection Weights')

file_config = {
                'NperE':NperE,
                'numEnsembles':num_ensembles,
                'dim':dim,
                'DperE': DperE,
                'cleanupN': cleanup_n,
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


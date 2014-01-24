
import time
overall_start = time.time()
import nengo
from nengo.matplotlib import rasterplot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='Droid Serif')

import argparse
from mytools import hrr, nf, fh, nengo_stack_plot
import random
import itertools

from build import build_learning_cleanup, build_cleanup_oja, build_cleanup_pes

seed = 810100000
random.seed(seed)

sim_class = nengo.Simulator
training_time = 2 #in seconds
testing_time = 0.5

DperE = 32
dim = 32
num_ensembles = int(dim / DperE)
dim = num_ensembles * DperE

neurons_per_vector = 20
#neurons_per_vector = 40
num_vectors = 5
cleanup_n = neurons_per_vector * num_vectors

NperD = 30
NperE = NperD * DperE
total_n = NperE * num_ensembles

max_rates=[400]
intercept=[0.1]
radius=1.0

pre_max_rates=[400] * NperE
#pre_radius=np.true_divide(1, np.sqrt(num_ensembles))
pre_intercepts=[0.1] * NperE
pre_radius=1.0
pre_ensemble_params = {'radius':pre_radius,
                       'max_rates':pre_max_rates,
                       'intercepts':pre_intercepts}

oja_scale = np.true_divide(5,1)
oja_learning_rate = np.true_divide(1,50)
pre_tau = 0.03
post_tau = 0.03
pes_learning_rate = np.true_divide(1,1)

vocab = hrr.Vocabulary(dim)
training_vectors = [vocab.parse("x"+str(i)).v for i in range(num_vectors)]
print "Training Vector Similarities:"
simils = []

if num_vectors > 1:
    for a,b in itertools.combinations(training_vectors, 2):
        s = np.dot(a,b)
        simils.append(s)
        print s
    print "Mean"
    print np.mean(simils)
    print "Max"
    print np.max(simils)
    print "Min"
    print np.min(simils)

noise = nf.make_hrr_noise(dim, 1)
testing_vectors = [noise(tv) for tv in training_vectors] + [hrr.HRR(dim).v]

gens = [nf.output(100, True, tv, False) for tv in training_vectors]
gens += [nf.output(100, True, tv, False) for tv in testing_vectors]
times = [training_time] * len(training_vectors) + [testing_time] * len(testing_vectors)
address_func = nf.make_f(gens, times)
storage_func = address_func

print "Building..."
start = time.time()

model = nengo.Model("Learn cleanup", seed=seed)

# ----- Make Input -----
address_input = nengo.Node(output=address_func)
storage_input = nengo.Node(output=storage_func)

# ----- Build neural part -----
#cleanup = build_training_cleanup(dim, num_vectors, neurons_per_vector, intercept=intercept)
cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n),
                      dimensions=dim,
                      max_rates=max_rates  * cleanup_n, intercepts=intercept * cleanup_n,
                      radius=radius)

pre_ensembles, pre_decoders, pre_connections = \
        build_cleanup_oja(model, address_input, cleanup, DperE, NperD, num_ensembles,
                          pre_ensemble_params, oja_learning_rate, oja_scale,
                          end_time=training_time * num_vectors)

output_ensembles, error_ensembles = build_cleanup_pes(cleanup, storage_input, DperE, NperD, num_ensembles, pes_learning_rate)

gate = nengo.Node(output=lambda x: [1.0] if x > training_time * num_vectors else [0.0])
for ens in error_ensembles:
    nengo.Connection(gate, ens.neurons, transform=-10 * np.ones((NperE, 1)))

# ----- Build probes -----
address_input_p = nengo.Probe(address_input, 'output')
storage_input_p = nengo.Probe(storage_input, 'output')
pre_probes = [nengo.Probe(ens, 'decoded_output', filter=0.1) for ens in pre_ensembles]
cleanup_s = nengo.Probe(cleanup, 'spikes')
output_probes =  [nengo.Probe(ens, 'decoded_output', filter=0.1) for ens in output_ensembles]

end = time.time()
print "Time:", end - start

# ----- Run and get data-----
print "Simulating..."
start = time.time()

sim = nengo.Simulator(model, dt=0.001)
sim.run(sum(times))

end = time.time()
print "Time:", end - start

# ----- Plot! -----
print "Plotting..."
start = time.time()

t = sim.trange()

num_plots = 6
offset = num_plots * 100 + 10 + 1

ax, offset = nengo_stack_plot(offset, t, sim, address_input_p, label='Input')
ax, offset = nengo_stack_plot(offset, t, sim, pre_probes, label='Pre')
ax, offset = nengo_stack_plot(offset, t, sim, cleanup_s, label='Cleanup Spikes')
ax, offset = nengo_stack_plot(offset, t, sim, output_probes, label='Output')

def make_sim_func(h):
    def sim(vec):
        return h.compare(hrr.HRR(data=vec))
    return sim

sim_funcs = [make_sim_func(hrr.HRR(data=h)) for h in training_vectors]
ax, offset = nengo_stack_plot(offset, t, sim, output_probes, label='Output',
                              func=sim_funcs)
ax, offset = nengo_stack_plot(offset, t, sim, address_input_p, label='Output',
                              func=sim_funcs)
file_config = {
                'NperE':NperE,
                'numEnsembles':num_ensembles,
                'dim':dim,
                'DperE': DperE,
                'premaxr':max_rates[0],
                'preint':pre_intercepts[0],
                'int':intercept,
                'ojascale':oja_scale,
                'lr':oja_learning_rate,
              }

filename = fh.make_filename('learning_cleanup', directory='learning_cleanup',
                            config_dict=file_config, extension='.png')
plt.savefig(filename)

end = time.time()
print "Time:", end - start

overall_end = time.time()
print "Total time: ", overall_end - overall_start

plt.show()


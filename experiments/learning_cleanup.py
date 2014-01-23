
import time
overall_start = time.time()
import nengo
from nengo.matplotlib import rasterplot
from nengo_ocl.sim_ocl import Simulator as SimOCL
from nengo_ocl.sim_npy import Simulator as SimNumpy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='Droid Serif')

import argparse
from mytools import hrr, nf, fh, nengo_stack_plot
import random
import itertools

from build import build_learning_cleanup, build_cleanup_oja, build_cleanup_pes

seed = 8101000
random.seed(seed)

sim_class = nengo.Simulator
learning_time = 2 #in seconds

DperE = 64
dim = 64
num_ensembles = int(dim / DperE)
dim = num_ensembles * DperE

neurons_per_vector = 40
num_vectors = 2
cleanup_n = neurons_per_vector * num_vectors

NperD = 30
NperE = NperD * DperE
total_n = NperE * num_ensembles

max_rates=[200]
intercept=[0.1]
radius=1.0

pre_max_rates=[400] * NperE
#pre_radius=np.true_divide(1, np.sqrt(num_ensembles))
pre_intercepts=[0.1] * NperE
pre_radius=1.0
pre_ensemble_params = {'radius':pre_radius,
                       'max_rates':pre_max_rates,
                       'intercepts':pre_intercepts}

oja_scale = np.true_divide(20,1)
oja_learning_rate = np.true_divide(1,50)
pre_tau = 0.03
post_tau = 0.03
#oja_scale = np.true_divide(10,1)
#oja_learning_rate = np.true_divide(1,20)
pes_learning_rate = np.true_divide(1,5)


vocab = hrr.Vocabulary(dim)
input_vectors = [vocab.parse("x"+str(i)).v for i in range(num_vectors)]
print "Input Vector Similarities:"
simils = []

if num_vectors > 1:
    for a,b in itertools.combinations(input_vectors, 2):
        s = np.dot(a,b)
        simils.append(s)
        print s
    print "Mean"
    print np.mean(simils)
    print "Max"
    print np.max(simils)
    print "Min"
    print np.min(simils)

gens = [nf.output(100, True, iv, False) for iv in input_vectors]
times = [learning_time] * num_vectors
input_func = nf.make_f(gens, times)

print "Building..."
start = time.time()

model = nengo.Model("Learn cleanup", seed=seed)

# ----- Make Input -----
inn = nengo.Node(output=input_func)

# ----- Build neural part -----
#cleanup = build_learning_cleanup(dim, num_vectors, neurons_per_vector, intercept=intercept)
cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n),
                      dimensions=dim,
                      max_rates=max_rates  * cleanup_n, intercepts=intercept * cleanup_n,
                      radius=radius)

pre_ensembles, pre_decoders, pre_connections = \
        build_cleanup_oja(model, inn, cleanup, DperE, NperD, num_ensembles,
                          pre_ensemble_params, oja_learning_rate, oja_scale)

output_ensembles, error_ensembles = build_cleanup_pes(cleanup, inn, DperE, NperD, num_ensembles, pes_learning_rate)

# ----- Build probes -----
inn_p = nengo.Probe(inn, 'output')
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

num_plots = 4
offset = num_plots * 100 + 10 + 1

ax, offset = nengo_stack_plot(offset, t, sim, inn_p, label='Input')
ax, offset = nengo_stack_plot(offset, t, sim, pre_probes, label='Pre')
ax, offset = nengo_stack_plot(offset, t, sim, cleanup_s, label='Cleanup Spikes')
ax, offset = nengo_stack_plot(offset, t, sim, output_probes, label='Output')

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


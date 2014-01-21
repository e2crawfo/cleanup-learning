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

DperE = 2
dim = 8
num_ensembles = int(dim / DperE)
dim = num_ensembles * DperE

cleanup_n = 20
NperD = 30
NperE = NperD * DperE

seed = 123
random.seed(seed)

learning_rate = np.true_divide(1,5)
input_vector = hrr.HRR(dim).v
cleanup_encoders = np.array([input_vector] * cleanup_n)

print "Building..."
start = time.time()
model = nengo.Model("Network Array PES", seed=seed)

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

#Plot data
print "Plotting..."
start = time.time()
t = sim.trange()

num_plots = 4
offset = num_plots * 100 + 10 + 1
plt.subplots_adjust(right=0.98, top=0.98, left=0.1, bottom=0.05, hspace=0.01)

ax, offset = nengo_stack_plot(offset, t, sim, input_probe, label='Input')
ax, offset = nengo_stack_plot(offset, t, sim, output_probes, label='Output')
sim_func = lambda x: np.dot(x, input_vector)
ax, offset = nengo_stack_plot(offset, t, sim, output_probes, func=sim_func, label='Similarity')
ax, offset = nengo_stack_plot(offset, t, sim, error_probes, label='Error', removex=False)

file_config = {}
#                'NperE':NperE,
#                'numEnsembles':num_ensembles,
#                'dim':dim,
#                'DperE': DperE,
#                'cleanupN': cleanup_n,
#                'premaxr':max_rates[0],
#                'preint':pre_intercepts[0],
#                'int':intercepts[0],
#                'ojascale':oja_scale,
#                'lr':oja_learning_rate,
#                'hrrnum':hrr_num,
#              }

filename = fh.make_filename('network_array_pes', directory='network_array_pes',
                            config_dict=file_config, extension='.png')
plt.savefig(filename)

end = time.time()
print "Plot Time: ", end - start

overall_end = time.time()
print "Total time: ", overall_end - overall_start

plt.show()



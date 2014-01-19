#import resource
#resource.setrlimit(resource.RLIMIT_AS, (megs * 1048576L, -1L))

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

parser = argparse.ArgumentParser(description='Learn a cleanup memory')

parser.add_argument('-s', default=1, type=int, help='Seed for rng')
parser.add_argument('-r', action='store_true', default=False, help='Supply to train on HRR')
parser.add_argument('--sim', default=0, type=int, help='0 for nengo.Simulator, 1 for sim_ocl, 2 for sim_npy')
parser.add_argument('--trial-length', dest='trial_length', default=2, type=int, help='Length of each trial/presentation (in seconds)')

parser.add_argument('-D', default=2, type=int, help='Number of dimensions')
parser.add_argument('--nperd', default=50, type=int, help='Number of neurons per dimension in other populations')
parser.add_argument('--cleanup-n', dest='cleanup_n', default=10, type=int, help='Number of neurons in cleanup population')
parser.add_argument('--vector-n', dest='vector_n', default=5, type=int, help='Number of neurons in cleanup population')
parser.add_argument('--cleanup-nperv', dest='cleanup_nperv', default=0, type=int, help='Number of neurons in cleanup population')


parser.add_argument('--oja-learning-rate', dest='oja_learning_rate', 
                    default=1e-3, type=float, help='OJA learning rate, on synapses entering cleanup population')
parser.add_argument('--no-oja', dest='no_oja', action='store_true', default=False, help='Supply to turn off OJA learning')

parser.add_argument('--pes-learning-rate', dest='pes_learning_rate',
                    default=1e1, type=float, help='PES learning rate, on synapses leaving cleanup population')
parser.add_argument('--no-pes', dest='no_pes', action='store_true', default=False, help='Supply to turn off PES learning')
parser.add_argument('--pes-scale', dest='pes_scale', default=0.1, type=float, help='Multiplier on transform between cleanup and post')

parser.add_argument('--post-filter', dest='post_filter', default=0.02, type=float, help='Filter on the connection between the cleanup and post')


argvals = parser.parse_args()

seed = argvals.s
np.random.seed(seed)

dim = argvals.D
n_per_d = argvals.nperd

do_oja = not argvals.no_oja
oja_learning_rate=argvals.oja_learning_rate

do_pes = not argvals.no_pes
pes_learning_rate=argvals.pes_learning_rate
pes_scale = argvals.pes_scale

max_rates=[200]
intercepts=[0.35]
percent_shown=1

ensemble_n = dim * n_per_d
vector_n = argvals.vector_n

if argvals.cleanup_nperv:
    cleanup_nperv = argvals.cleanup_nperv
else:
    cleanup_nperv = np.true_divide(argvals.cleanup_n, argvals.vector_n)
    cleanup_nperv = np.ceil(cleanup_nperv)

cleanup_n = cleanup_nperv * vector_n

post_filter = argvals.post_filter

trial_length = argvals.trial_length #in seconds
plot_connection_weights = 0
train_on_hrr = argvals.r

sim_class = [nengo.Simulator, SimOCL, SimNumpy][argvals.sim]

def make_input(encoders):
    times = []
    times.extend([trial_length for i in range(vector_n)])
    times.extend([trial_length for i in range(vector_n)])

    gens = []
    if train_on_hrr:
        gens.extend([nf.output(100, False, pe, False, nf.make_hrr_noise(dim, 1)) for pe in encoders])
    else:
        gens.extend([nf.output(100, True, pe, False) for pe in encoders])
    gens.extend([nf.output(100, False, pe, False, nf.make_hrr_noise(dim, 1)) for pe in encoders])

    return times, gens

# ----- Building Model -----
model = nengo.Model("Learn Cleanup", seed=seed)

#We make the connection matrix between pre and cleanup out of the decoders and encoders.
#This gives us control over the initial sparsity of the representation in the cleanup

#Get decoders
pre = nengo.Ensemble(label='pre', neurons=nengo.LIF(ensemble_n), dimensions=dim,
                     intercepts=[0.1] * ensemble_n)
pre_p = nengo.Probe(pre, 'decoded_output')
sim = nengo.Simulator(model, dt=0.001)
sim.run(.01)
pre_decoders = sim.model.connections[0]._decoders

#Make up our own encoders
vocab = hrr.Vocabulary(dim, max_similarity=0.05)
cleanup_encoders = []
for i in range(vector_n):
    cleanup_encoders.append(vocab.parse("x" + str(i)).v)

cleanup_encoders = np.array(cleanup_encoders)
times, gens = make_input(cleanup_encoders)
cleanup_encoders = np.repeat(cleanup_encoders, cleanup_nperv, 0)

# --- Make Input Function
sim_length = sum(times)
func = nf.make_f(gens, times)

# ----- Make Nodes -----
inn = nengo.Node(output=func)

cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n), dimensions=dim,
                      max_rates=max_rates  * cleanup_n, intercepts=intercepts * cleanup_n, encoders=cleanup_encoders)

post = nengo.Ensemble(label='post', neurons=nengo.LIF(ensemble_n), dimensions=dim,
                      #max_rates=max_rates  * ensemble_n, 
                      intercepts=[-.1] * ensemble_n)

error = nengo.Ensemble(label='Error', neurons=nengo.LIF(ensemble_n), dimensions=dim)

# ----- Make Connections -----
nengo.Connection(inn, pre)
nengo.Connection(inn, error)
nengo.Connection(post, error, transform=np.eye(dim) * -1.0)

# OJA Rule
if do_oja:
    oja_rule = OJA(learning_rate=oja_learning_rate)
    transform = np.dot(cleanup_encoders, pre_decoders)
    conn = nengo.Connection(pre.neurons, cleanup.neurons,
                        transform=transform, learning_rule=oja_rule)
else:
    conn = nengo.Connection(pre, cleanup)

# PES Rule - Start with small weights going out of the cleanup population
pes_rule = PES(error) if do_pes else None
nengo.Connection(cleanup, post, transform=np.eye(dim) * pes_scale, learning_rule=pes_rule, filter=post_filter)

# ----- Make Probes -----
inn_p = nengo.Probe(inn, 'output')
pre_s = nengo.Probe(pre, 'spikes')
cleanup_p = nengo.Probe(cleanup, 'decoded_output', filter=0.05)
post_p = nengo.Probe(post, 'decoded_output', filter=0.1)
error_p = nengo.Probe(error, 'decoded_output')

slices = np.index_exp[:]
cleanup_s = nengo.Probe(cleanup, 'spikes', slices=slices)

slices = np.index_exp[0:1,:]
if do_oja:
    weights_p = nengo.Probe(conn, 'transform', slices=slices)

if plot_connection_weights:
    oja_weights = nengo.Probe(oja_conn, 'transform')

# ----- Run Model -----
print "Simulating for", sim_length, " seconds."
sim = sim_class(model, dt=0.001)
sim.run(sim_length)

# ---- Plotting -----
t = sim.trange()
trunc = int(percent_shown * len(t))
spike_time = t[:trunc]
t = t[-trunc:]

num_plots = 6 + int(plot_connection_weights) * min(1, cleanup_n)
offset = num_plots * 100 + 10 + 1
def remove_xlabels():
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticklabels([])
plt.subplots_adjust(right=0.98, top=0.98, left=0.1, bottom=0.05, hspace=0.01)

plt.subplot(offset)
plt.plot(t, sim.data(inn_p)[-trunc:,:], label='Input')
plt.ylabel('Input')
remove_xlabels()
offset += 1

plt.subplot(offset)
plt.plot(t, sim.data(post_p)[-trunc:,:], label='Post')
plt.ylabel('Post: Decoded')
remove_xlabels()
offset += 1

plt.subplot(offset)
plt.plot(t, sim.data(error_p)[-trunc:,:], label='Error')
plt.ylabel('Error: Decoded')
remove_xlabels()
offset += 1

plt.subplot(offset)
plt.plot(t, sim.data(cleanup_p)[-trunc:,:], label='Cleanup')
plt.ylabel('Cleanup: (Orig. Decoders)')
remove_xlabels()
offset += 1

plt.subplot(offset)
rasterplot(spike_time, sim.data(cleanup_s)[-trunc:,:])
plt.ylabel('Cleanup: Spikes')
offset += 1

plt.subplot(offset)
connection_weights = np.squeeze(sim.data(weights_p)[-trunc:,:,:])
maxes = np.amax(connection_weights, 1)
mins = np.amin(connection_weights, 1)
plt.plot(t, maxes, label='weights')
plt.plot(t, mins, label='weights')
plt.ylabel('Oja Weights')
offset += 1

if plot_connection_weights:
    extremes_only = True
    for i in range(min(1, cleanup_n)):
        plt.subplot(offset)
        connection_weights = sim.data(oja_weights)[-trunc:,i,:]
        connection_weights = np.squeeze(connection_weights)
        if extremes_only:
            maxes = np.amax(connection_weights, 1)
            mins = np.amin(connection_weights, 1)
            plt.plot(t, maxes, label='cleanup, filter=0.1')
            plt.plot(t, mins, label='cleanup, filter=0.1')
        else:
            plt.plot(t, connection_weights, label='cleanup, filter=0.1')
        remove_xlabels()
        offset += 1



file_config = {
                'ensembleN':ensemble_n,
                'cleanupNperV':cleanup_nperv,
                'vectorN':vector_n,
                'dim':dim,
                'trainonhrr':train_on_hrr,
              }

filename = fh.make_filename('cleanup', directory='plots',
                            config_dict=file_config, extension='.png')
plt.savefig(filename)
plt.show()



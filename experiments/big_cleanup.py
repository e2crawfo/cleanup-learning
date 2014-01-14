#learning with oja!
#import resource
#resource.setrlimit(resource.RLIMIT_AS, (megs * 1048576L, -1L))

import nengo
from nengo.plot import rasterplot
from nengo_ocl.sim_ocl import Simulator as SimOCL
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mytools import hrr, nf

parser = argparse.ArgumentParser(description='Learn a cleanup memory')
parser.add_argument('-s', default=1, type=int, help='Seed for rng')
parser.add_argument('-r', default=1, type=int, help='Whether to train on HRR')
argvals = parser.parse_args()

seed = argvals.s
np.random.seed(seed)

dim = 32
learning_rate=1e-3
max_rates=[200]
intercepts=[0.4]
percent_shown=1

pre_n = 500
post_n = 100

trial_length=2
plot_connection_weights = 0
train_on_hrr=argvals.r

times = []
times.extend([trial_length * 3 for i in range(post_n)])
times.extend([trial_length * 3 for i in range(post_n)])
sim_length = sum(times)

vocab = hrr.Vocabulary(dim, max_similarity=0.05)
post_encoders = []
for i in range(post_n):
    post_encoders.append(vocab.parse("x" + str(i)).v)

post_encoders = np.array(post_encoders)

gens = []
if train_on_hrr:
    gens.extend([nf.output(100, False, pe, False, nf.make_hrr_noise(dim, 1)) for pe in post_encoders])
else:
    gens.extend([nf.output(100, True, pe, False) for pe in post_encoders])
gens.extend([nf.output(100, False, pe, False, nf.make_hrr_noise(dim, 1)) for pe in post_encoders])

func = nf.make_f(gens, times)


model = nengo.Model("OJA", seed=seed)
pre = nengo.Ensemble(label='pre', neurons=nengo.LIF(pre_n), dimensions=dim,
                     intercepts=[0.0] * pre_n)

#run for 1 second to get the decoders
pre_p = nengo.Probe(pre, 'decoded_output')
sim = nengo.Simulator(model, dt=0.001)
sim.run(1)
pre_decoders = sim.model.connections[0]._decoders


# ----- Building Model -----
inn = nengo.Node(output=func)

post = nengo.Ensemble(label='post', neurons=nengo.LIF(post_n), dimensions=dim,
                      max_rates=max_rates  * post_n, intercepts=intercepts * post_n)
nengo.Connection(inn, pre)

transform = np.dot(post_encoders, pre_decoders)
for t in transform:
    print np.linalg.norm(t)

conn = nengo.Connection(pre.neurons, post.neurons,
                 transform=transform,
                 )

# ----- Making Probes -----
inn_p = nengo.Probe(inn, 'output')

pre_s = nengo.Probe(pre, 'spikes')
post_s = nengo.Probe(post, 'spikes')

if plot_connection_weights:
    oja_weights = nengo.Probe(oja_conn, 'transform')


# ----- Running Model -----
sim = SimOCL(model, dt=0.001)
sim.run(sim_length)


# ---- Plotting -----
t = sim.trange()
trunc = int(percent_shown * len(t))
spike_time = t[:trunc]
t = t[-trunc:]

num_plots = 2 + plot_connection_weights * min(1, post_n)
offset = num_plots * 100 + 10 + 1
def remove_xlabels():
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticklabels([])
plt.subplots_adjust(right=0.98, top=0.98, left=0.05, bottom=0.05, hspace=0.01)

plt.subplot(offset)
plt.plot(t, sim.data(inn_p)[-trunc:,:], label='Input')
remove_xlabels()
offset += 1

if plot_connection_weights:
    extremes_only = True
    for i in range(min(1, post_n)):
        plt.subplot(offset)
        connection_weights = sim.data(oja_weights)[-trunc:,i,:]
        connection_weights = np.squeeze(connection_weights)
        if extremes_only:
            maxes = np.amax(connection_weights, 1)
            mins = np.amin(connection_weights, 1)
            plt.plot(t, maxes, label='post, filter=0.1')
            plt.plot(t, mins, label='post, filter=0.1')
        else:
            plt.plot(t, connection_weights, label='post, filter=0.1')
        remove_xlabels()
        offset += 1

plt.subplot(offset)
rasterplot(spike_time, sim.data(post_s)[-trunc:,:])
offset += 1

filename = 'big_oja_pre_%s_post_%s_dim_%s_trainonhrr_%s_intercept_%s.pdf' % (str(pre_n), str(post_n),
            str(dim), str(train_on_hrr), str(intercepts[0]))
plt.savefig(filename)



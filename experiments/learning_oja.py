#learning with oja!

import nengo
from nengo.nonlinearities import OJA
from nengo.plot import rasterplot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import hrr
import noise_functions as nf

similarities = []

parser = argparse.ArgumentParser(description='Learn a cleanup memory')
parser.add_argument('-s', default=1, type=int, help='Seed for rng')
argvals = parser.parse_args()

seed = argvals.s
np.random.seed(seed)

dim = 10
sim_length = 10
tau_theta = 2
learning_rate=1e-1
max_rates=[200]
intercepts=[0.25]
percent_shown=1

pre_n = 400

trial_length=200
tick = 0

D = dim
vocab = hrr.Vocabulary(D)
v = vocab.parse("A")
main_vector = v.v

func = nf.make_f( [nf.output(200, True, main_vector, False),
                   nf.output(200, False, main_vector, False),
                   nf.output(200, False, main_vector, False),
                   nf.output(200, True, main_vector, True, nf.make_hrr_noise(dim, 3)),
                   ],
                   [10,20,30])


model = nengo.Model("OJA", seed=seed)
pre = nengo.Ensemble(label='pre', neurons=nengo.LIF(pre_n), dimensions=dim,
                     intercepts=[0.0] * pre_n)
pre_p = nengo.Probe(pre, 'decoded_output')
sim = nengo.Simulator(model, dt=0.001)
sim.run(1)

#decoders = sim.model.connections[0]._decoders
#encoders = sim.model.objs[0].encoders
#print encoders.shape
#print np.linalg.norm(encoders[0, :])
#print decoders.shape


inn = nengo.Node(output=func)

post = nengo.Ensemble(label='post', neurons=nengo.LIF(1), dimensions=dim,
                      max_rates=max_rates, intercepts=intercepts)
nengo.Connection(inn, pre)

learning_rule=OJA(pre_tau=0.05, post_tau=0.05, learning_rate=learning_rate)

oja_conn = nengo.Connection(pre.neurons, post.neurons,
                 transform=1*np.ones((post.n_neurons, pre.n_neurons)),
                 learning_rule=learning_rule
                 )

inn_p = nengo.Probe(inn, 'output')

pre_s = nengo.Probe(pre, 'spikes')
post_s = nengo.Probe(post, 'spikes')

oja_weights = nengo.Probe(oja_conn, 'transform')
oja_oja = nengo.Probe(learning_rule, 'oja')
oja_pre = nengo.Probe(learning_rule, 'pre')
oja_post = nengo.Probe(learning_rule, 'post')
oja_delta = nengo.Probe(learning_rule, 'delta')

sim = nengo.Simulator(model, dt=0.001)
sim.run(sim_length)

t = sim.trange()
trunc = int(percent_shown * len(t))
spike_time = t[:trunc]
t = t[-trunc:]

num_plots = 6
offset = num_plots * 100 + 10
def remove_xlabels():
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticklabels([])
plt.subplots_adjust(right=0.98, top=0.98, left=0.05, bottom=0.05, hspace=0.01)

plt.subplot(offset + 1)
plt.plot(t, sim.data(inn_p)[-trunc:,:], label='Input')
#plt.plot(t, sim.data(pre_p), label='pre, filter=0.1')
#plt.plot(t, sim.data(post_p), label='post, filter=0.1')
#plt.legend(loc=0, prop={'size': 10})
remove_xlabels()

plt.subplot(offset + 2)
rasterplot(spike_time, sim.data(pre_s)[-trunc:,:])
remove_xlabels()

plt.subplot(offset + 3)
rasterplot(spike_time, sim.data(post_s)[-trunc:,:])
remove_xlabels()

plt.subplot(offset + 4)
connection_weights = sim.data(oja_weights)[-trunc:,:,:]
connection_weights = np.squeeze(connection_weights)
plt.plot(t, connection_weights, label='post, filter=0.1')
remove_xlabels()

plt.subplot(offset + 5)
print sim.data(oja_oja).shape
sim_oja = sim.data(oja_oja)[-trunc:, :, :]
sim_oja = np.squeeze(sim_oja)
plt.plot(t, sim_oja, label='post, filter=0.1')
remove_xlabels()

#plt.subplot(offset + 6)
#plt.plot(t, sim.data(oja_pre), label='post, filter=0.1')
#remove_xlabels()

#plt.subplot(offset + 7)
#plt.plot(t, sim.data(oja_post), label='post, filter=0.1')
#remove_xlabels()

plt.subplot(offset + 6)
delta = sim.data(oja_delta)[-trunc:,:,:]
delta = np.squeeze(delta)
plt.plot(t, delta, label='post, filter=0.1')

plt.show()


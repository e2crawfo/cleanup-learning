#learning with bcm!

import nengo
from nengo.nonlinearities import BCM
from nengo.plot import rasterplot
import numpy as np
import matplotlib.pyplot as plt
import argparse
import hrr

similarities = []

parser = argparse.ArgumentParser(description='Learn a cleanup memory')
parser.add_argument('-s', default=1, type=int, help='Seed for rng')
argvals = parser.parse_args()

seed = argvals.s
np.random.seed(seed)

dim = 512
sim_length = 40
tau_theta = 2
learning_rate=1e-1
max_rates=[200]
intercepts=[0.25]
percent_shown=1

pre_n = 200

trial_length=200
tick = 0

D = dim
vocab = hrr.Vocabulary(D)
v = vocab.parse("A")
main_vector = v.v

alternate = False
flip = False

def default_noise(vec):
    vector = np.random.rand(dim) - 0.5
    vector = vector / np.linalg.norm(vector)
    return vector

def flip_noise(vec):
    return -vec

import random
def make_hrr_noise(D, num):
    def hrr_noise(input_vec):
        noise_vocab = hrr.Vocabulary(D)
        keys = [noise_vocab.parse(str(x)) for x in range(2*num+1)]

        input_vec = hrr.HRR(data=input_vec)
        partner_key = random.choice(keys)

        pair_keys = filter(lambda x: x != partner_key, keys)

        pairs = random.sample(pair_keys, 2 * num)
        p0 = (pairs[x] for x in range(0,len(pairs),2))
        p1 = (pairs[x] for x in range(1,len(pairs),2))
        S = map(lambda x, y: noise_vocab[x].convolve(noise_vocab[y]), p0, p1)

        S = reduce(lambda x, y: x + y, S, noise_vocab[partner_key].convolve(input_vec))
        S.normalize()

        vec_hrr = S.convolve(~noise_vocab[partner_key])
        similarity = vec_hrr.compare(input_vec)
        similarities.append(similarity)
        return vec_hrr.v
    return hrr_noise

def output(trial_length, main, main_vector, alternate, noise_func=default_noise):
    tick = 0
    vector = main_vector
    main_hrr = hrr.HRR(data=main_vector)

    while True:
        if tick == trial_length :
            tick = 0
            if main:
                vector = main_vector
            else:
                vector = noise_func(main_vector)
                u = hrr.HRR(data=vector)
                similarity = u.compare(main_hrr)
                print "Sim:", similarity

            if alternate:
                main = not main

        tick += 1

        yield vector


def make_f(generators, times):
    generators
    def f(t):
        if len(generators) > 1 and t > times[0]:
            generators.pop(0)
            times.pop(0)
        return generators[0].next()
    return f

func = make_f( [#output(100, True, main_vector, False, False),
                output(200, True, main_vector, False),
                #output(100, True, main_vector, True),
                output(200, False, main_vector, False),
                output(200, False, main_vector, False),
                output(200, True, main_vector, True, make_hrr_noise(dim, 3)),
                #output(100, True, main_vector, True),
                #output(100, True, main_vector, True, make_hrr_noise(dim, 1)),
                #output(1000, False, main_vector, False, False),
                #output(200, False, main_vector, True, False),
                ],
                [10,20,30])

                #[5,10,15,20])#[4, 10])


model = nengo.Model("BCM", seed=seed)

inn = nengo.Node(output=func)
pre = nengo.Ensemble(label='pre', neurons=nengo.LIF(pre_n), dimensions=dim,
                     intercepts=[0.0] * pre_n)

post = nengo.Ensemble(label='post', neurons=nengo.LIF(1), dimensions=dim,
                      max_rates=max_rates, intercepts=intercepts)
nengo.Connection(inn, pre)

learning_rule=BCM(tau_theta, pre_tau=0.05, post_tau=0.05,
                  learning_rate=learning_rate, label="BCM")

bcm_conn = nengo.Connection(pre.neurons, post.neurons,
                 transform=1*np.ones((post.n_neurons, pre.n_neurons)),
                 learning_rule=learning_rule
                 )

inn_p = nengo.Probe(inn, 'output')

pre_s = nengo.Probe(pre, 'spikes')
post_s = nengo.Probe(post, 'spikes')

bcm_weights = nengo.Probe(bcm_conn, 'transform')
bcm_theta = nengo.Probe(learning_rule, 'theta')
bcm_pre = nengo.Probe(learning_rule, 'pre')
bcm_post = nengo.Probe(learning_rule, 'post')
bcm_delta = nengo.Probe(learning_rule, 'delta')

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
connection_weights = sim.data(bcm_weights)[-trunc:,:,:]
connection_weights = np.squeeze(connection_weights)
plt.plot(t, connection_weights, label='post, filter=0.1')
remove_xlabels()

plt.subplot(offset + 5)
plt.plot(t, sim.data(bcm_theta)[-trunc:, :], label='post, filter=0.1')
remove_xlabels()

#plt.subplot(offset + 6)
#plt.plot(t, sim.data(bcm_pre), label='post, filter=0.1')
#remove_xlabels()

#plt.subplot(offset + 7)
#plt.plot(t, sim.data(bcm_post), label='post, filter=0.1')
#remove_xlabels()

plt.subplot(offset + 6)
delta = sim.data(bcm_delta)[-trunc:,:,:]
delta = np.squeeze(delta)
plt.plot(t, delta, label='post, filter=0.1')

plt.show()


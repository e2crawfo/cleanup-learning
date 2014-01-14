#lateral inhibition!

import nengo
from nengo.nonlinearities import OJA
from nengo.plot import rasterplot
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse
import hrr
import noise_functions as nf

similarities = []

parser = argparse.ArgumentParser(description='A network with lateral inhibition')
parser.add_argument('-s', default=1, type=int, help='Seed for rng')
argvals = parser.parse_args()

seed = argvals.s
np.random.seed(seed)

dim = 2
sim_length = 1 
tau_theta = 2
learning_rate=1e-1
max_rates=[200]
intercepts=[0.25]
percent_shown=1

pre_n = 10
inhib_n = 21

trial_length=200
tick = 0

D = dim
amp1 = 4
amp2 = 30
bump_width = 1


noise = nf.make_f([nf.output(200, False, np.zeros(dim), True)], [])

model = nengo.Model("Lateral Inhibition", seed=seed)

inn = nengo.Node(output=noise)
pre = nengo.Ensemble(label='pre', neurons=nengo.LIF(pre_n), dimensions=dim,
                     intercepts=[0.0] * pre_n)
inhib = nengo.Ensemble(label='inhib', neurons=nengo.LIF(inhib_n), dimensions=dim,
                     intercepts=[0.05] * inhib_n)

nengo.Connection(inn, pre)

#feedforward
nengo.Connection(pre.neurons, inhib.neurons,
                 #transform=0.001*np.random.rand(inhib.n_neurons, pre.n_neurons),
                 transform=0.001*np.ones((inhib.n_neurons, pre.n_neurons)),
                 )

def make_mexican_hat(sigma):
    def mexican_hat(x):
        return 0.5 * (4 * np.exp(-x**2 / sigma**2) - np.exp(-x**2/(10.0 * sigma**2)))
    return mexican_hat

def make_box(s, A=1.0, B=0.0):
    def box(x):
        if abs(x) > s:
            return B
        else:
            return A
    return box

g1 = norm(loc=0, scale=4).pdf
g2 = norm(loc=0, scale=20).pdf
breadth = int(np.floor(inhib_n / 2))
eval_points = np.arange(-breadth, breadth+1, 1)

weight_func = lambda y: amp1 * g1(y) - 0.14
weight_func = make_mexican_hat(2.0)
weight_func = make_box(3, 0.05, -0.05)
weight_func = lambda x: 0.0
#weight_func = lambda y: amp1 * g1(y) -  amp2 * g2(y)

x = np.array(map(weight_func, eval_points))
x[breadth] = 0.0
connections = x
x = np.roll(x, -breadth)
transform = []
for i in range(inhib_n):
    row = np.roll(x, i) 
    transform.append(row)

transform = np.array(transform)

#recurrent
nengo.Connection(inhib.neurons, inhib.neurons,
                 transform=transform,
                 )


inn_p = nengo.Probe(inn, 'output')
pre_s = nengo.Probe(pre, 'spikes')
inhib_s = nengo.Probe(inhib, 'spikes')

sim = nengo.Simulator(model, dt=0.001)
sim.run(sim_length)

t = sim.trange()
trunc = int(percent_shown * len(t))
spike_time = t[:trunc]
t = t[-trunc:]

num_plots = 4
offset = num_plots * 100 + 10
def remove_xlabels():
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticklabels([])
plt.subplots_adjust(right=0.98, top=0.98, left=0.05, bottom=0.05, hspace=0.1)

plt.subplot(offset + 1)
plt.plot(t, sim.data(inn_p)[-trunc:,:], label='Input')
remove_xlabels()

plt.subplot(offset + 2)
rasterplot(spike_time, sim.data(pre_s)[-trunc:,:])
remove_xlabels()

plt.subplot(offset + 3)
rasterplot(spike_time, sim.data(inhib_s)[-trunc:,:])
remove_xlabels()

plt.subplot(offset + 4)
plt.plot(eval_points, connections)

plt.show()



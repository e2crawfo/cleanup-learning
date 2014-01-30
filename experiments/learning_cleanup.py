
import time
overall_start = time.time()
import nengo
from nengo.matplotlib import rasterplot

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set(font='Droid Serif')

import argparse
from mytools import hrr, nf, fh, timed
from mytools import extract_probe_data, nengo_plot_helper
import random
import itertools

from build import build_learning_cleanup, build_cleanup_oja, build_cleanup_pes


@timed.namedtimer("build_and_run_vectors")
def build_and_run_vectors(seed, dim, DperE, NperD, num_vectors, neurons_per_vector, training_time,
                          testing_time, cleanup_params, ensemble_params, oja_learning_rate,
                          oja_scale, pre_tau, post_tau, pes_learning_rate, **kwargs):

    cleanup_n = neurons_per_vector * num_vectors

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

    noise = nf.make_hrr_noise(dim, 2)
    testing_vectors = [noise(tv) for tv in training_vectors] + [hrr.HRR(dim).v]

    build_and_run_(seed, dim, DperE, NperD, cleanup_n, training_vectors, testing_vectors,
                      training_time, testing_time, cleanup_params, ensemble_params,
                      oja_learning_rate, oja_scale, pre_tau, post_tau, pes_learning_rate)


@timed.namedtimer("build_and_run")
def build_and_run(seed, dim, DperE, NperD, cleanup_n, training_vectors, testing_vectors,
                  training_time, testing_time, cleanup_params, ensemble_params, oja_learning_rate,
                  oja_scale, pre_tau, post_tau, pes_learning_rate, **kwargs):

    random.seed(seed)

    num_ensembles = int(dim / DperE)
    dim = num_ensembles * DperE

    NperE = NperD * DperE
    total_n = NperE * num_ensembles

    ensemble_params['max_rates'] *= NperE
    ensemble_params['intercepts'] *= NperE

    cleanup_params['max_rates'] *= cleanup_n
    cleanup_params['intercepts'] *= cleanup_n

    gens = [nf.output(100, True, tv, False) for tv in training_vectors]
    gens += [nf.output(100, True, tv, False) for tv in testing_vectors]
    times = [training_time] * len(training_vectors) + [testing_time] * len(testing_vectors)
    address_func = nf.make_f(gens, times)
    storage_func = address_func

    print "Building..."

    model = nengo.Model("Learn cleanup", seed=seed)

    # ----- Make Input -----
    address_input = nengo.Node(output=address_func)
    storage_input = nengo.Node(output=storage_func)

    # ----- Build neural part -----
    #cleanup = build_training_cleanup(dim, num_vectors, neurons_per_vector, intercept=intercept)
    cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n),
                          dimensions=dim, **cleanup_params)

    pre_ensembles, pre_decoders, pre_connections = \
            build_cleanup_oja(model, address_input, cleanup, DperE, NperD, num_ensembles,
                              ensemble_params, oja_learning_rate, oja_scale,
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

    # ----- Run and get data-----
    print "Simulating..."

    sim = nengo.Simulator(model, dt=0.001)
    sim.run(sum(times))

    return locals()


@timed.namedtimer("extract_data")
def extract_data(filename, sim, address_input_p, storage_input_p,
                 pre_probes, cleanup_s, output_probes, sim_funcs, **kwargs):

    t = sim.trange()
    address_input, _ = extract_probe_data(t, sim, address_input_p)
    pre_decoded, _ = extract_probe_data(t, sim, pre_probes)
    cleanup_spikes, _ = extract_probe_data(t, sim, cleanup_s, spikes=True)
    output_decoded, _ = extract_probe_data(t, sim, output_probes)
    output_sim, _ = extract_probe_data(t, sim, output_probes, func=sim_funcs)
    input_sim, _ = extract_probe_data(t, sim, address_input_p, func=sim_funcs)

    ret = dict(t=t, address_input=address_input, pre_decoded=pre_decoded,
               cleanup_spikes=cleanup_spikes, output_decoded=output_decoded,
               output_sim=output_sim, input_sim=input_sim)

    fh.npsave(filename, **ret)

    return ret


@timed.namedtimer("plot")
def plot(filename, t, address_input, pre_decoded, cleanup_spikes,
          output_decoded, output_sim, input_sim, **kwargs):

    num_plots = 6
    offset = num_plots * 100 + 10 + 1

    ax, offset = nengo_plot_helper(offset, t, address_input)
    ax, offset = nengo_plot_helper(offset, t, pre_decoded)
    ax, offset = nengo_plot_helper(offset, t, cleanup_spikes, spikes=True)
    ax, offset = nengo_plot_helper(offset, t, output_decoded)
    ax, offset = nengo_plot_helper(offset, t, output_sim)
    ax, offset = nengo_plot_helper(offset, t, input_sim)

    plt.savefig(filename)

def start():
    seed = 81223

    training_time = 1 #in seconds
    testing_time = 0.5

    DperE = 32
    dim = 32
    NperD = 30

    neurons_per_vector = 20
    num_vectors = 5

    oja_scale = np.true_divide(2,1)
    oja_learning_rate = np.true_divide(1,50)
    pre_tau = 0.03
    post_tau = 0.03
    pes_learning_rate = np.true_divide(1,1)

    cleanup_params = {'radius':1.0,
                       'max_rates':[400],
                       'intercepts':[0.13]}

    ensemble_params = {'radius':1.0,
                       'max_rates':[400],
                       'intercepts':[0.1]}

    config = locals()

    #intercepts actually matter quite a bit
    config['cint'] = cleanup_params['intercepts'][0]
    config['eint'] = ensemble_params['intercepts'][0]

    do_plots = True

    data_title = 'lcdata'
    directory = 'learning_cleanup_data'

    data_filename = fh.make_filename(data_title, directory=directory,
                                     config_dict=config, extension='.npz',
                                     use_time=False,
                                     omit=['ensemble_params', 'cleanup_params'])

    data = fh.npload(data_filename)

    if data is None:
        results = build_and_run_vectors(**config)

        sim = results['sim']
        training_vectors = results['training_vectors']

        def make_sim_func(h):
            def sim(vec):
                return h.compare(hrr.HRR(data=vec))
            return sim
        sim_funcs = [make_sim_func(hrr.HRR(data=h)) for h in training_vectors]
        data = extract_data(filename=data_filename, sim_funcs=sim_funcs, **results)

    if do_plots:
        plot_title = 'lcplot'
        directory='learning_cleanup_plots'
        plot_filename = fh.make_filename(plot_title, directory=directory,
                                    config_dict=config, extension='.png',
                                    omit=['ensemble_params', 'cleanup_params'])
        plot(filename=plot_filename, **data)
        plt.show()

if __name__ == "__main__":
    start()


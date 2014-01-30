
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

    ret = build_and_run(seed, dim, DperE, NperD, cleanup_n, training_vectors, training_vectors,
                   testing_vectors, training_time, testing_time, cleanup_params, ensemble_params,
                   oja_learning_rate, oja_scale, pre_tau, post_tau, pes_learning_rate)

    return ret


@timed.namedtimer("build_and_run")
def build_and_run(seed, dim, DperE, NperD, cleanup_n, address_vectors, stored_vectors,
                  testing_vectors, training_time, testing_time, cleanup_params, ensemble_params,
                  oja_learning_rate, oja_scale, pre_tau, post_tau, pes_learning_rate, **kwargs):

    random.seed(seed)

    num_ensembles = int(dim / DperE)
    dim = num_ensembles * DperE

    NperE = NperD * DperE
    total_n = NperE * num_ensembles

    ensemble_params['max_rates'] *= NperE
    ensemble_params['intercepts'] *= NperE

    cleanup_params['max_rates'] *= cleanup_n
    cleanup_params['intercepts'] *= cleanup_n


    address_gens = [nf.output(100, True, av, False) for av in address_vectors]
    address_gens += [nf.output(100, True, tv, False) for tv in testing_vectors]
    stored_gens = [nf.output(100, True, sv, False) for sv in stored_vectors]
    stored_gens += [nf.output(100, True, tv, False) for tv in testing_vectors]

    address_times = [training_time] * len(address_vectors) + [testing_time] * len(testing_vectors)
    stored_times = [training_time] * len(address_vectors) + [testing_time] * len(testing_vectors)
    address_func = nf.make_f(address_gens, address_times)
    stored_func = nf.make_f(stored_gens, stored_times)

    sim_time = sum(address_times)

    end_time = len(address_vectors) * training_time

    print "Building..."

    model = nengo.Model("Learn cleanup", seed=seed)

    # ----- Make Input -----
    address_input = nengo.Node(output=address_func)
    stored_input = nengo.Node(output=stored_func)

    # ----- Build neural part -----
    #cleanup = build_training_cleanup(dim, num_vectors, neurons_per_vector, intercept=intercept)
    cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n),
                          dimensions=dim, **cleanup_params)

    pre_ensembles, pre_decoders, pre_connections = \
            build_cleanup_oja(model, address_input, cleanup, DperE, NperD, num_ensembles,
                              ensemble_params, oja_learning_rate, oja_scale,
                              end_time=end_time)

    output_ensembles, error_ensembles = build_cleanup_pes(cleanup, stored_input, DperE, NperD, num_ensembles, pes_learning_rate)

    gate = nengo.Node(output=lambda x: [1.0] if x > end_time else [0.0])
    for ens in error_ensembles:
        nengo.Connection(gate, ens.neurons, transform=-10 * np.ones((NperE, 1)))

    # ----- Build probes -----
    address_input_p = nengo.Probe(address_input, 'output')
    stored_input_p = nengo.Probe(stored_input, 'output')
    pre_probes = [nengo.Probe(ens, 'decoded_output', filter=0.1) for ens in pre_ensembles]
    cleanup_s = nengo.Probe(cleanup, 'spikes')
    output_probes =  [nengo.Probe(ens, 'decoded_output', filter=0.1) for ens in output_ensembles]

    # ----- Run and get data-----
    print "Simulating..."

    sim = nengo.Simulator(model, dt=0.001)
    sim.run(sim_time)

    return locals()


@timed.namedtimer("extract_data")
def extract_data(filename, sim, address_input_p, stored_input_p,
                 pre_probes, cleanup_s, output_probes,
                 address_vectors, stored_vectors, **kwargs):

    t = sim.trange()
    address_input, _ = extract_probe_data(t, sim, address_input_p)
    stored_input, _ = extract_probe_data(t, sim, stored_input_p)
    pre_decoded, _ = extract_probe_data(t, sim, pre_probes)
    cleanup_spikes, _ = extract_probe_data(t, sim, cleanup_s, spikes=True)
    output_decoded, _ = extract_probe_data(t, sim, output_probes)

    def make_sim_func(h):
        def sim(vec):
            return h.compare(hrr.HRR(data=vec))
        return sim

    address_sim_funcs = [make_sim_func(hrr.HRR(data=h)) for h in address_vectors]
    stored_sim_funcs = [make_sim_func(hrr.HRR(data=h)) for h in stored_vectors]

    output_sim, _ = extract_probe_data(t, sim, output_probes, func=stored_sim_funcs)
    input_sim, _ = extract_probe_data(t, sim, address_input_p, func=address_sim_funcs)

    ret = dict(t=t, address_input=address_input, stored_input=stored_input,
               pre_decoded=pre_decoded, cleanup_spikes=cleanup_spikes,
               output_decoded=output_decoded,
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

    config = locals()

    cleanup_params = {'radius':1.0,
                       'max_rates':[400],
                       'intercepts':[0.13]}

    ensemble_params = {'radius':1.0,
                       'max_rates':[400],
                       'intercepts':[0.1]}

    #intercepts actually matter quite a bit
    config['cint'] = cleanup_params['intercepts'][0]
    config['eint'] = ensemble_params['intercepts'][0]

    do_plots = True

    data_title = 'lcdata'
    directory = 'learning_cleanup_data'

    data_filename = fh.make_filename(data_title, directory=directory,
                                     config_dict=config, extension='.npz',
                                     use_time=False)

    data = fh.npload(data_filename)

    if data is None:
        results = build_and_run_vectors(ensemble_params=ensemble_params,
                                        cleanup_params=cleanup_params,
                                        **config)
        data = extract_data(filename=data_filename, **results)

    if do_plots:
        plot_title = 'lcplot'
        directory='learning_cleanup_plots'
        plot_filename = fh.make_filename(plot_title, directory=directory,
                                    config_dict=config, extension='.png')
        plot(filename=plot_filename, **data)
        plt.show()

if __name__ == "__main__":
    start()


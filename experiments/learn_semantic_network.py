import matplotlib.pyplot as plt
import numpy as np
import random
from mytools import hrr, timed
from build_semantic_network import build_semantic_network
from learning_cleanup import build_and_run

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
    N = 10

    oja_scale = np.true_divide(2,1)
    oja_learning_rate = np.true_divide(1,50)
    pre_tau = 0.03
    post_tau = 0.03
    pes_learning_rate = np.true_divide(1,1)

    config = locals()

    #Don't put all parematers in config
    cleanup_params = {'radius':1.0,
                       'max_rates':[400],
                       'intercepts':[0.13]}

    ensemble_params = {'radius':1.0,
                       'max_rates':[400],
                       'intercepts':[0.1]}

    #intercepts actually matter quite a bit
    config['cint'] = cleanup_params['intercepts'][0]
    config['eint'] = ensemble_params['intercepts'][0]


    data_title = 'lsndata'
    directory = 'learning_sn_data'

    data_filename = fh.make_filename(data_title, directory=directory,
                                     config_dict=config, extension='.npz',
                                     use_time=False)

    data = fh.npload(data_filename)

    if data is None:
        #build the graph and get the vectors encoding it
        hrr_vectors, id_vectors, edge_vectors = build_semantic_network(dim, N, seed=seed)

        config['training_vectors'] = []
        config['testing_vectors'] = []

        results = build_and_run(**config)

        sim = results['sim']
        training_vectors = results['training_vectors']

        data = extract_data(filename=data_filename, sim_funcs=sim_funcs, **results)

    do_plots = True
    if do_plots:
        plot_title = 'lsnplot'
        directory='learning_sn_plots'
        plot_filename = fh.make_filename(plot_title, directory=directory,
                                    config_dict=config, extension='.png')
        plot(filename=plot_filename, **data)
        plt.show()


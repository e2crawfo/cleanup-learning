import nengo
from nengo.nonlinearities import OJA, PES
from cutilities import minimum_threshold

import numpy as np

def build_learning_cleanup(dim, num_vectors, neurons_per_vector,
                           intercept=None, radius=1.0, max_rate=200):
    cleanup_n = neurons_per_vector * num_vectors

    if intercept is None:
        prob, intercept = minimum_threshold(0.9, neurons_per_vector, cleanup_n, dim)

    print "Threshold:", intercept
    cleanup = nengo.Ensemble(label='cleanup',
                             neurons=nengo.LIF(cleanup_n),
                             dimensions=dim,
                             radius=radius,
                             max_rates=[max_rate]  * cleanup_n,
                             intercepts=[intercept] * cleanup_n,
                             )
    return cleanup

def build_cleanup_oja(model, inn, cleanup, DperE, NperD, num_ensembles,
                      ensemble_params, learning_rate, oja_scale, encoders=None,
                      pre_decoders=None, use_oja=True, pre_tau=0.03, post_tau=0.03,
                      end_time=None):

    NperE = NperD * DperE
    dim = DperE * num_ensembles

    # ----- Make Nodes -----
    pre_ensembles = []
    for i in range(num_ensembles):
        pre_ensembles.append(nengo.Ensemble(label='pre_'+str(i), neurons=nengo.LIF(NperE),
                            dimensions=DperE,
                            **ensemble_params))
        #intercepts=pre_intercepts * NperE,
        #                    max_rates=pre_max_rates * NperE,
        #                    radius=pre_radius))

    # ----- Get decoders for pre populations. We use them to initialize the connection weights
    if pre_decoders is None:
        dummy = nengo.Ensemble(label='dummy',
                                neurons=nengo.LIF(NperE),
                                dimensions=dim)

        def make_func(dim, start):
            def f(x):
                y = np.zeros(dim)
                y[start:start+len(x)] = x
                return y
            return f

        for i, pre in enumerate(pre_ensembles):
            nengo.Connection(pre, dummy, function=make_func(dim, i * DperE))

    if encoders is None or pre_decoders is None:
        sim = nengo.Simulator(model, dt=0.001)
        sim.run(.01)

    if pre_decoders is None:
        pre_decoders = {}
        for conn in sim.model.connections:
            if conn.pre.label.startswith('pre'):
                pre_decoders[conn.pre.label] = conn._decoders

    if encoders is None:
        for obj in sim.model.objs:
            if obj.label.startswith('cleanup'):
                encoders = obj.encoders
                break

    # ----- Make Connections -----
    pre_connections = []

    in_transform=np.eye(DperE)
    in_transform = np.concatenate((in_transform, np.zeros((DperE, dim - DperE))), axis=1)
    for pre in pre_ensembles:

        nengo.Connection(inn, pre, transform=in_transform)
        in_transform = np.roll(in_transform, DperE, axis=1)

        connection_weights = np.dot(encoders, pre_decoders[pre.label])
        connection_weights *= np.true_divide(1, cleanup.radius)

        oja_rule = None
        if use_oja:
            oja_rule = OJA(pre_tau=pre_tau, post_tau=post_tau,
                            learning_rate=learning_rate, oja_scale=oja_scale,
                            end_time=end_time)

        conn = nengo.Connection(pre.neurons, cleanup.neurons,
                                transform=connection_weights, learning_rule=oja_rule)
        pre_connections.append(conn)

    return pre_ensembles, pre_decoders, pre_connections


def build_cleanup_pes(cleanup, error_input, DperE, NperD, num_ensembles, learning_rate):

    NperE = NperD * DperE
    dim = DperE * num_ensembles

    # ----- Make Nodes -----
    output_ensembles=[]
    for i in range(num_ensembles):
        ens = nengo.Ensemble(label='output'+str(i), neurons=nengo.LIF(NperE),
                        dimensions=DperE,
                        )

        output_ensembles.append(ens)

    error_ensembles=[]
    for i in range(num_ensembles):
        ens = nengo.Ensemble(label='error'+str(i), neurons=nengo.LIF(NperE),
                        dimensions=DperE,
                        )

        error_ensembles.append(ens)

    # ----- Make Connections -----
    def make_func(lo,hi):
        def f(x):
            #return -x[lo:hi] 
            return [0] * (hi - lo)
        return f

    transform=np.eye(DperE)
    transform = np.concatenate((transform, np.zeros((DperE, dim - DperE))), axis=1)

    for i, o, e in zip(range(num_ensembles), output_ensembles, error_ensembles):
        lo = i * DperE
        hi = (i + 1) * DperE
        f = make_func(lo, hi)
        pes_rule = PES(e, learning_rate = learning_rate)
        nengo.Connection(cleanup, o, function=f,
                learning_rule=pes_rule)

        nengo.Connection(o, e, transform=np.eye(DperE) * -1)

        nengo.Connection(error_input, e, transform=transform)
        transform = np.roll(transform, DperE, axis=1)

    return output_ensembles, error_ensembles


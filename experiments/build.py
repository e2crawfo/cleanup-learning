import nengo
from nengo.nonlinearities import OJA, PES

import numpy as np

def build_learning_cleanup(model, dim, num_vectors, neurons_per_vector, **kwargs):
    with model:
        cleanup = nengo.Ensemble(label='cleanup',
                                 neurons=nengo.LIF(cleanup_n),
                                 dimensions=dim,
                                 max_rates=max_rates  * cleanup_n,
                                 intercepts=intercepts * cleanup_n,
                                 encoders=encoder)
        return cleanup



def build_cleanup_oja(model, inn, cleanup, **kwargs):
    with model:

        # ----- Make Nodes -----

        pre_ensembles = []
        for i in range(num_ensembles):
            pre_ensembles.append(nengo.Ensemble(label='pre_'+str(i), neurons=nengo.LIF(npere),
                                dimensions=DperE,
                                intercepts=pre_intercepts * npere,
                                max_rates=pre_max_rates * npere,
                                radius=pre_radius))

        # ----- Get decoders for pre populations. We use them to initialize the connection weights
        dummy = nengo.Ensemble(label='dummy',
                                neurons=nengo.LIF(npere),
                                dimensions=dim)

        def make_func(dim, start):
            def f(x):
                y = np.zeros(dim)
                y[start:start+len(x)] = x
                return y
            return f

        for i, pre in enumerate(pre_ensembles):
            nengo.Connection(pre, dummy, function=make_func(dim, i * DperE))

        sim = nengo.Simulator(model, dt=0.001)
        sim.run(.01)

        pre_decoders = {}
        for conn in sim.model.connections:
            if conn.pre.label.startswith('pre'):
                pre_decoders[conn.pre.label] = conn._decoders

        # ----- Make Connections -----

        in_transform=np.eye(DperE)
        in_transform = np.concatenate((in_transform, np.zeros((DperE, dim - DperE))), axis=1)
        for pre in pre_ensembles:

            nengo.Connection(inn, pre, transform=in_transform)
            in_transform = np.roll(in_transform, DperE, axis=1)

            oja_rule = OJA(pre_tau=0.05, post_tau=0.05,
                    learning_rate=oja_learning_rate, oja_scale=oja_scale)
            connection_weights = np.dot(encoder, pre_decoders[pre.label])
            conn = nengo.Connection(pre.neurons, cleanup.neurons,
                                    transform=connection_weights, learning_rule=oja_rule)

        return pre_ensembles


def build_cleanup_pes(model, cleanup, error_input, DperE, NperD, num_ensembles, learning_rate):

    NperE = NperD * DperE
    dim = DperE * num_ensembles

    with model:
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


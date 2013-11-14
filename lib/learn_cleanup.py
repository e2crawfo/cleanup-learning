__author__ = 'e2crawfo'

import nef
import hrr

from nef.simplenode import SimpleNode
import nef.templates.learned_termination as learning
import nef.templates.gate as gating

import random
import itertools as it
import numeric as np
import logging

"""
Dimension will be specified by user.
Then the remaining parameters are:
number of cleanup neurons
probability that each clean vector has at least n representative neurons
threshold on the neurons

These ones are all interlinked, and specifying any two of them determines
the third.
"""

class ExperimentController(SimpleNode):

    input_vector = None
    learning_vector = None
    learning_gate = 1.0

    is_learning = True
    never_switch = False

    def __init__(self, name, vocab, num_vectors, clean_learning, learning_noise, testing_noise,
                 learning_bias, testing_bias, trial_length,
                 schedule_func=None, user_control_learning=False):

        self.logger = logging.getLogger("ExperimentController")
        self.logger.info("__init__")

        self.num_vectors = num_vectors
        self.vocab = vocab
        self.generate_vectors()

        self.user_control_learning = user_control_learning

        self.trial_length = trial_length #in timesteps
        self.clean_learning = clean_learning

        self.noise = None
        self.learning_noise = learning_noise
        self.testing_noise = testing_noise

        self.bias = 0.0
        self.testing_bias = testing_bias
        self.learning_bias = learning_bias

        self.schedule_func = schedule_func
        self.epoch_length = 1


        self.error_history = []

        SimpleNode.__init__(self, name)

    def init(self):

        self.logger.info("init")

        self.results = []

        self.learning_input = [1.0]
        self.never_switch = False
        self.test_schedule=self.schedule_func()

        self.is_learning = False
        self.switch_phase()

        self.setup_trial()

        self.trial_error = []

    def origin_input_vecs(self):
        return self.input_vector

    def origin_learning_vecs(self):
        return self.learning_vector

    def origin_learning_control(self):
        return [self.learning_gate]

    def origin_bias(self):
        return [self.bias]

    def termination_learning_on(self, x):
        self.learning_input = x

    def noisy_vector(self, index):
        v = self.noise(self.vectors[index])
        return v

    def record_trial_error(self):
        #throw out samples at the beginning of the trial
        self.trial_error = self.trial_error[20:]
        squared_error = [np.sum(te**2) for te in self.trial_error]
        if len(squared_error) != 0:
            self.logger.info("RMSE for trial: %g" % (np.sqrt(np.mean(squared_error))))
            print "RMSE for trial: %g" % np.sqrt(np.mean(squared_error))

        if not self.is_learning:
            self.error_history.extend(squared_error)

        self.trial_error = []

    def setup_trial(self):
        next_vector_index = random.choice(range(self.num_vectors))
        self.input_vector = self.noisy_vector(next_vector_index)
        self.correct_vector = self.vectors[next_vector_index]

        if self.clean_learning:
            self.learning_vector = self.vectors[next_vector_index]
        else:
            self.learning_vector = self.input_vector

        self.time_step_index = 0

        self.logger.info("Setup trial. After: Vector Index: %s" % next_vector_index)

    def switch_phase(self):

        #switch from learning to testing or vice versa
        if self.user_control_learning:
            self.is_learning = self.learning_input[0] > 0.5
        else:
            if self.never_switch:
                return
            try:
                self.epoch_length = self.test_schedule.next()
                self.is_learning = not self.is_learning
            except StopIteration:
                self.never_switch = True
                return

        if self.is_learning:
            self.learning_gate = 1.0
            self.noise = self.learning_noise
            self.bias = self.learning_bias
        else:
            self.learning_gate = 0.0
            self.noise = self.testing_noise
            self.bias = self.testing_bias

        self.presentations = 0

        self.log_string = ("Switching phase. After: Learning: %s,"
                           " Learning Gate: %g, Bias: %g")

        self.logger.info(self.log_string % (str(self.is_learning), self.learning_gate, self.bias))

    def tick(self):

        self.time_step_index += 1
        end_trial = self.time_step_index == self.trial_length

        if end_trial:
            self.record_trial_error()

            self.presentations += 1
            if self.presentations == self.epoch_length:
                self.switch_phase()

            #set vectors for the next trial
            self.setup_trial()

    def reset(self, randomize=False):
        SimpleNode.reset(self, randomize)
        self.calc_stats()

    def calc_stats(self):

        if len(self.error_history) != 0:
            self.latest_rmse = np.sqrt(np.mean(self.error_history))
            self.logger.info("Latest RMSE: %g" % (self.latest_rmse))

        self.error_history = []
        self.trial_error = []

    def generate_vectors(self, replace_vocab=False):
        num_keys = [0]
        for key in self.vocab.keys:
            try:
                x = int('0' + key, 16)
                num_keys.append(x)
            except:
                continue
        max_key = max(num_keys)

        names = [hex(i + max_key + 1)[1:] for i in range(self.num_vectors)]
        vectors = [self.vocab.parse(name) for name in names]
        for v in vectors:
            v.normalize()

        self.vectors = [v.v for v in vectors]
        self.logger.info("Generated vectors: %s" % (names))
        sims = []
        for i in range(len(names)):
            name1 = names[i]
            hrr1 = self.vocab.parse(name1)

            for j in range(i):
                name2 = names[j]
                hrr2 = self.vocab.parse(name2)
                similarity = hrr1.compare(hrr2)
                sims.append(similarity)
                self.logger.info("Similarity between %s and %s: %g" % (name1, name2, similarity))

        self.logger.info("Similarity stats: Mean %g, Max %g, Min %g" \
                % (np.mean(sims), max(sims), min(sims)))

def make_bias(net, name, gated, bias, neurons, pstc=0.001, direct=False):

    if direct:
        gate=net.make(name, 1, 1, mode='direct')
    else:
        gate=net.make(name, neurons, 1, mode='default')

    #net.connect(gate, None, func=addOne, origin_name='bias', create_projection=False)

    output=net.network.getNode(gated)

    weights=[[bias]]*output.neurons

    tname='bias'
    output.addTermination(tname, weights, pstc, False)

    orig = gate.getOrigin('X')
    term = output.getTermination(tname)
    net.network.addProjection(orig, term)

def make_learnable_cleanup(D=16, cleanup_neurons = 1000, num_vecs = 4, threshold=(-0.9,0.9), max_rate=(100,200),
                           radius=1.0, cleanup_pstc=0.001, neurons_per_dim=50, clean_learning=False,
                           trial_length=100, learning_noise=0.6, testing_noise=0.3,
                           user_control_learning=False, user_control_bias=False, learning_rate=5e-6,
                           schedule_func=None, neural_input=False, learning_bias=0.0, testing_bias=0.0, **kwargs):
    """
    Construct a cleanup memory that initially has no vocabulary, and learns its vocabulary from the vectors
    it receives as input. Also constructs an experiment node that tests the cleanup. Should probably separate
    that part out eventually.

    :param variable_bias: For using different thresholds during learning and testing. Implemented by a
    decoded-to-nondecoded connection from an ensemble to the cleanup population.

    :type variable_bias: boolean or tuple. If a tuple, first value will be used as threshold during learning,
    second will be used as threshold during testing. If True, user controls threshold (only works with a GUI). If False,
    threshold is fixed at whatever is determined by t_hi and t_lo.

    """

    print cleanup_neurons
    logger = logging.getLogger("make_learnable_cleanup")

    net = nef.Network('learn_cleanup', seed=2)

    vocab=hrr.Vocabulary(D)

    func_str = "def tr(self, x, dimensions=%d, pstc=0.02):\n   self.results = x\n   self.trial_error.append(self.correct_vector - self.results)" % D
    exec func_str in locals()
    ExperimentController.termination_results = tr

    controller = ExperimentController('EC', vocab, num_vecs, clean_learning, learning_noise,
                    testing_noise, learning_bias, testing_bias, trial_length,
                    schedule_func=schedule_func, user_control_learning=user_control_learning)

    net.add(controller)

    logger.info("Adding cleanup")

    net.make('cleanup', neurons=cleanup_neurons, dimensions=D, radius=radius, intercept=threshold, max_rate=max_rate, tau_ref=0.004)

    if user_control_bias:
        logger.info("Adding bias controlled by user")
        make_bias(net, 'bias', 'cleanup', bias=1.0, neurons=1, pstc=cleanup_pstc, direct=True)
        net.make_input('bias_input', [0])
        net.connect('bias_input', 'bias')
    else:
        logger.info("Adding bias controlled by EC")
        make_bias(net, 'bias', 'cleanup', bias=1.0, neurons=1, pstc=cleanup_pstc, direct=True)
        net.connect(controller.getOrigin('bias'), 'bias')

    logger.info("Adding output")
    net.make('output', neurons=neurons_per_dim * D, dimensions=D, mode='default')

    logger.info("Adding input")
    if neural_input:
        net.make_array('input', neurons=neurons_per_dim, length=D, dimensions=1, mode='default')
    else:
        net.make('input', neurons=1, dimensions=D, mode='direct')

    net.connect( controller.getOrigin('input_vecs'), 'input')
    net.connect('input', 'cleanup', pstc=cleanup_pstc)


    logger.info("Adding error population and learning")
    learning.make(net, errName = 'error', N_err = neurons_per_dim * D, preName='cleanup', postName='output', rate=learning_rate)
    net.connect( controller.getOrigin('learning_vecs'), 'error', pstc=0.01)

    logger.info("Adding learning gate")
    gating.make(net, name='Gate', gated='error', neurons=40, pstc=0.01)

    net.connect('output', controller.getTermination('results'))

    if user_control_learning:
        logger.info("Adding learning-control switch")
        net.make_input('switch', [1.0])
        net.connect('switch', controller.getTermination('learning_on'))

    net.connect( controller.getOrigin('learning_control'), 'Gate')

    logger.info("Adding network to nengo")
    net.add_to_nengo()

    #if show_stats:
    #    encoders = net.get('cleanup').getEncoders()

    #    sims=[[] for name in names]
    #    hrrs = [vocab.parse(name) for name in names]

    #    for enc in encoders:
    #        h = hrr.HRR(data=enc)

    #        for v, s in zip(hrrs,sims):
    #            s.append(v.compare(h))

    #        sims.append(s)

    #    for v,s, in zip(hrrs,sims):
    #        print "lo"
    #        print len(filter(lambda x: x > t_lo, s))
    #        print "hi"
    #        print len(filter(lambda x: x > t_hi, s))

    #    print sims

    return net


def replace_cleanup(net, learning_rate=5e-5, threshold=0.0, radius=1.0, max_rate=(100,200), cleanup_pstc=0.001, post_term="cleanup_00"):

    cleanup = net.get('cleanup')
    cleanup_neurons = cleanup.neurons
    D = cleanup.dimension

    try:
        bias_termination = cleanup.getTermination(u'bias')
        bias_node = net.get('bias')
        bias_weights = bias_termination.getNodeTerminations()[0].weights[0]
        bias_pstc = bias_termination.tau
        has_bias = True
    except:
        has_bias = False

    net.remove('cleanup')
    output = net.get('output')

    term = output.getTermination(post_term)
    net.network.removeProjection(term)

    # random weight matrix to initialize projection from pre to post
    def rand_weights(w):
        for i in range(len(w)):
            for j in range(len(w[0])):
                w[i][j] = random.uniform(-1e-3,1e-3)
        return w
    weight = rand_weights(np.zeros((output.neurons, cleanup.neurons)).tolist())
    term.setTransform(weight, False)

    cleanup = net.make('cleanup', neurons=cleanup_neurons, dimensions=D, radius=radius, intercept=threshold, max_rate=max_rate, tau_ref=0.004)

    net.connect(cleanup.getOrigin('AXON'), term)
    net.connect('input', 'cleanup', pstc=cleanup_pstc)

    if has_bias:
        weights=[[bias_weights]]*cleanup_neurons

        tname='bias'
        cleanup.addTermination(tname, weights, bias_pstc, False)

        orig = bias_node.getOrigin('X')
        term = cleanup.getTermination(tname)
        net.network.addProjection(orig, term)



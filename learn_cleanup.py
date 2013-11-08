__author__ = 'e2crawfo'

import cleanup_lib.cleanup_utilities as cu
import nef
import hrr
from stats.bootstrapci import bootstrapci

from nef.simplenode import SimpleNode
import nef.templates.learned_termination as learning
import nef.templates.gate as gating
from ca.nengo.util.impl import NodeThreadPool

from optparse import OptionParser
import random
import itertools as it
import numeric as np
import ConfigParser
import logging
import sys

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

    is_training = True
    never_switch = False

    dt = None

    def __init__(self, name, vocab, num_vectors, clean_learning, training_noise, testing_noise,
                 trial_length, schedule_func=None, user_control_learning=False, 
                 variable_bias=None):

        self.logger = logging.getLogger("ExperimentController")
        self.logger.info("__init__")

        self.num_vectors = num_vectors
        self.vocab = vocab
        self.generate_vectors()

        self.training_noise = training_noise
        self.testing_noise = testing_noise
        self.trial_length = trial_length #in timesteps
        self.clean_learning = clean_learning

        self.schedule_func = schedule_func
        self.epoch_length = 1
        self.bias = 0.0
        self.user_control_learning = user_control_learning
        self.variable_bias = variable_bias
        self.error_history = []

        SimpleNode.__init__(self, name)

    def init(self):

        self.logger.info("init")
        self.dt = None

        self.results = []

        self.learning_input = [1.0]
        self.never_switch = False
        self.test_schedule=self.schedule_func()

        self.is_training = False
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
        if not self.is_training:
            #throw out samples at the beginning of the trial
            self.trial_error = self.trial_error[20:]
            squared_error = [np.sum(te**2) for te in self.trial_error]
            if len(squared_error) != 0:
                self.logger.info("RMSE for trial: %g" % (np.sqrt(np.mean(squared_error))))
            self.error_history.extend(squared_error)

        self.trial_error = []

    def set_bias(self):
        """
        self.is_training must be set before calling this for it to have the desired effectj
        """
        if type(self.variable_bias) == tuple:
            if self.is_training:
                self.bias = self.variable_bias[0]
            else:
                self.bias = self.variable_bias[1]

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
            self.is_training = self.learning_input[0] > 0.5
        else:
            if self.never_switch:
                return
            try:
                self.epoch_length = self.test_schedule.next()
                self.is_training = not self.is_training
            except StopIteration:
                self.never_switch = True
                return

        self.set_bias()

        if self.is_training:
            self.learning_gate = 1.0
            self.noise = self.training_noise
        else:
            self.learning_gate = 0.0
            self.noise = self.testing_noise

        self.presentations = 0

        self.log_string = ("Switching phase. After: Learning: %s,"
                           " Learning Gate: %g, Bias: %g")

        self.logger.info(self.log_string % (str(self.is_training), self.learning_gate, self.bias))

    def tick(self):

        if self.dt is None and self.t > 0.00001:
            self.dt = self.t

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

def make_bias(net, name, gated, bias, neurons, pstc=0.001, direct=False):

    if direct:
        gate=net.make(name, 1, 1, intercept=(-0.7, 0), encoders=[[-1]], mode='direct')
    else:
        gate=net.make(name, neurons, 1, intercept=(-0.7, 0), encoders=[[-1]], mode='default')

    #net.connect(gate, None, func=addOne, origin_name='bias', create_projection=False)

    output=net.network.getNode(gated)

    weights=[[bias]]*output.neurons

    tname='bias'
    output.addTermination(tname, weights, pstc, False)
    
    orig = gate.getOrigin('X')
    term = output.getTermination(tname)
    net.network.addProjection(orig, term)

def make_learnable_cleanup(D, cleanup_neurons = 1000, num_vecs = 4, t_lo=0.6, t_hi=1.0,
                           neurons_per_dim=50, clean_learning=False, trial_length=100,
                           training_noise=0.6,testing_noise=0.3,
                           user_control_learning=False, learning_rate=5e-6,
                           schedule_func=None, show_stats=False, 
                           use_neural_input=False, variable_bias=None):
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

    logger = logging.getLogger("make_learnable_cleanup")

    net = nef.Network('learn_cleanup', seed=2)

    vocab=hrr.Vocabulary(D)

    func_str = "def tr(self, x, dimensions=%d, pstc=0.02):\n   self.results = x\n   self.trial_error.append(self.correct_vector - self.results)" % D
    exec func_str in locals()
    ExperimentController.termination_results = tr

    controller = ExperimentController('EC', vocab, num_vecs, clean_learning, training_noise,
                    testing_noise, trial_length, schedule_func=schedule_func, 
                    user_control_learning=user_control_learning,
                    variable_bias=variable_bias)

    net.add(controller)

    logger.info("Adding cleanup")
    threshold_ratio = float(t_lo)/float(t_hi)
    net.make('cleanup', neurons=cleanup_neurons, dimensions=D, radius=t_hi, intercept=(threshold_ratio, threshold_ratio + 0.8 * (1 - threshold_ratio)))

    if type(variable_bias) == tuple:
        logger.info("Adding bias controlled by EC")
        make_bias(net, 'bias', 'cleanup', bias=1.0, neurons=1, direct=True)
        net.connect(controller.getOrigin('bias'), 'bias')
    elif variable_bias:
        logger.info("Adding bias controlled by user")
        make_bias(net, 'bias', 'cleanup', bias=1.0, neurons=1, direct=True)
        net.make_input('bias_input', [0])
        net.connect('bias_input', 'bias')

    logger.info("Adding output")
    net.make('output', neurons=neurons_per_dim * D, dimensions=D, mode='default')

    logger.info("Adding input")
    if use_neural_input:
        net.make_array('input', neurons=neurons_per_dim, length=D, dimensions=1, mode='default')
    else:
        net.make('input', neurons=1, dimensions=D, mode='direct')

    net.connect( controller.getOrigin('input_vecs'), 'input')
    net.connect('input', 'cleanup', pstc=0.001)


    logger.info("Adding error population and learning")
    learning.make(net, errName = 'error', N_err = neurons_per_dim * D, preName='cleanup', postName='output', rate=learning_rate)
    net.connect( controller.getOrigin('learning_vecs'), 'error', pstc=0.001)

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

    if show_stats:
        encoders = net.get('cleanup').getEncoders()

        sims=[[] for name in names]
        hrrs = [vocab.parse(name) for name in names]

        for enc in encoders:
            h = hrr.HRR(data=enc)

            for v, s in zip(hrrs,sims):
                s.append(v.compare(h))

            sims.append(s)

        for v,s, in zip(hrrs,sims):
            print "lo"
            print len(filter(lambda x: x > t_lo, s))
            print "hi"
            print len(filter(lambda x: x > t_hi, s))

        print sims

    return net


def replace_cleanup(net, D, cleanup_neurons, t_lo, t_hi, learning_rate):

    cleanup = net.get('cleanup')

    try:
        bias_termination = cleanup.getTermination(u'bias')
        bias_node = net.get('bias')
        bias_weights = bias_termination.getNodeTerminations()[0].weights[0]
        bias_pstc = bias_termination.tau
        has_bias = True
    except:
        has_bias = False

    net.remove('cleanup')
    net.remove('error')
    output = net.get('output')

    for t in output.getTerminations():
        name = t.getName()

        term = output.getTermination(name)
        net.network.removeProjection(term)

        try:
            output.removeTermination(name)
        except:
            output.removeDecodedTermination(name)

        #net.network.removeProjection(mod_00)

    threshold_ratio = float(t_lo)/float(t_hi)
    cleanup = net.make('cleanup', neurons=cleanup_neurons, dimensions=D, radius=t_hi, intercept=(threshold_ratio, threshold_ratio + 0.8 * (1 - threshold_ratio)))

    net.connect('input', 'cleanup', pstc=0.001)
    learning.make(net, errName = 'error', preName='cleanup', postName='output', rate=learning_rate)

    if has_bias:
        weights=[[bias_weights]]*cleanup_neurons

        tname='bias'
        cleanup.addTermination(tname, weights, bias_pstc, False)

        orig = bias_node.getOrigin('X')
        term = cleanup.getTermination(tname)
        net.network.addProjection(orig, term)

def make_simple_test_schedule(training, testing):
    def f():
        yield training
        yield testing
    return f

def simple_noise(noise):

    def f(input_vec):
        v = input_vec + noise * hrr.HRR(D).v
        v = v / sum(v**2)
        return v

    return f

def hrr_noise(D, num):

    noise_vocab = hrr.Vocabulary(D)
    keys = [noise_vocab.parse(str(x)) for x in range(2*num+1)]

    def f(input_vec):

        input_vec = hrr.HRR(data=input_vec)
        partner_key = random.choice(keys)

        pair_keys = filter(lambda x: x != partner_key, keys)

        pairs = random.sample(pair_keys, 2 * num)
        p0 = (pairs[x] for x in range(0,len(pairs),2))
        p1 = (pairs[x] for x in range(1,len(pairs),2))
        S = map(lambda x, y: noise_vocab[x].convolve(noise_vocab[y]), p0, p1)

        S = reduce(lambda x, y: x + y, S, noise_vocab[partner_key].convolve(input_vec))
        S.normalize()

        return S.convolve(~noise_vocab[partner_key]).v

    return f

if __name__=="__main__":
    parser = OptionParser()
    parser.add_option("-N", "--numneurons", default=800, type="int", help="Number of neurons in cleanup")
    parser.add_option("-n", "--neuronsperdim", default=50, type="int", help="Number of neurons per dimension for other ensembles")
    parser.add_option("-D", "--dim", default=16, type="int", help="Dimension of the vectors that the cleanup operates on")
    parser.add_option("-V", "--numvectors", default=4, type="int", help="Number of vectors that the cleanup will try to learn")
    parser.add_option("--dt", default=0.001, type="float", help="Time step")
    parser.add_option("-a", "--alpha", default=5e-5, type="float", help="Learning rate")
    parser.add_option("-L", "--triallength", default=100, type="int", help="Length of each vector presentation, in timesteps")
    parser.add_option("-P", "--learningpres", default=100, type="int", help="Number of presentations during learning")
    parser.add_option("-p", "--testingpres", default=20, type="int", help="Number of presentations during testing")
    parser.add_option("-T", "--learningnoise", default=0.5, type="float", help="Parameter for the noise during learning")
    parser.add_option("-t", "--testingnoise", default=0.5, type="float", help="Parameter for the noise during testing")
    parser.add_option("-R", "--numruns", default=0, type="int", help="Number of runs to do. We can reuse certain things between runs, so this speeds up the process of doing several runs at once")

    parser.add_option("--varbias", default="user", help="Whether to use different biases during learning and testing")
    parser.add_option("-B", "--learningbias", default=.4, type="float", help="Amount of bias during learning. Only has an effect if varthresh is True")
    parser.add_option("-b", "--testingbias", default=.3, type="float", help="Amount of bias during testing. Only has an effect if varthresh is True")

    parser.add_option("--Phi", default=.9, type="float", help="Probability for hi")
    parser.add_option("--Plo", default=.9, type="float", help="Probability for low")

    parser.add_option("--Vhi", default=10, type="int", help="Number of neurons for hi")
    parser.add_option("--Vlo", default=20, type="int", help="Number of neurons for low")

    parser.add_option("--threads", default=1, type="int", help="Number of threads to use to run the simulation")

    parser.add_option("--resultsfile", default="results", help="Name of file to write results to")
    parser.add_option("--logfile", default="log", help="Name of file to log to")

    parser.add_option("-c", "--cleanlearning", default=False, help="Whether to set decoders using clean vectors (supervised) or not (unsupervised)")
    parser.add_option("-I", "--neuralinput", default=False, help="Whether the input should be passed through a neural population first")
    parser.add_option("--replacecleanup", default=True, help="Whether to generate a new cleanup neural population for each run.")
    parser.add_option("--replacevectors", default=True, help="Whether to generate new vocabulary vectors for each run.")
    parser.add_option("--errorlearning", default=False, help="Whether to use error for learning (alternative is to use the learning vector as is).")
    parser.add_option("--dry-run", dest="dry_run", action='store_true', default=False, help="Whether to use error for learning (alternative is to use the learning vector as is).")

    (options, args) = parser.parse_args()
    print "options: ", options
    print "args: ", args

    logging.basicConfig(filename=options.logfile, filemode='w', level=logging.INFO)
    logging.info("Parameters: " + str(options) + str(args))

    command_line = 'cl' in args

    if options.dry_run:
        logging.info("Dry run!")
        config = ConfigParser.ConfigParser()

        config.add_section('Error')
        config.set('Error', 'mean', 0.0)
        config.set('Error', 'var', 0.0)

        f = open(options.resultsfile, 'w')
        config.write(f)
        f.close()
        sys.exit()

    D = options.dim
    N = options.numneurons

    neurons_per_dim = options.neuronsperdim

    num_vecs = options.numvectors

    training_noise_val = options.learningnoise
    training_noise = simple_noise(training_noise_val)
    #training_noise = hrr_noise(D, 1) 

    testing_noise_val = options.testingnoise
    testing_noise = simple_noise(testing_noise_val)
    #testing_noise = hrr_noise(D, 1)

    use_neural_input = options.neuralinput
    clean_learning = options.cleanlearning
    learning_rate = options.alpha
    trial_length = options.triallength
    schedule_func = make_simple_test_schedule(options.learningpres, options.testingpres)

    variable_bias = options.varbias
    if variable_bias == "user" and not command_line:
        variable_bias = True
    elif variable_bias:
        variable_bias = (options.learningbias, options.testingbias)

    P_hi = options.Phi
    P_lo = options.Plo

    V_hi = options.Vhi
    V_lo = options.Vlo

    _, threshold_lo = cu.minimum_threshold(P_lo, V_lo, N, D)
    _, threshold_hi = cu.minimum_threshold(P_hi, V_hi, N, D)

    user_control_learning = not command_line
    num_runs = options.numruns


    network = make_learnable_cleanup(D, N, num_vecs, threshold_lo, threshold_hi,
                    neurons_per_dim=neurons_per_dim, 
                    clean_learning=clean_learning, learning_rate=learning_rate,
                    training_noise=training_noise, testing_noise=testing_noise, 
                    schedule_func=schedule_func, trial_length=trial_length, 
                    user_control_learning=user_control_learning, 
                    use_neural_input=use_neural_input, 
                    variable_bias=variable_bias)

    NodeThreadPool.setNumJavaThreads(options.threads)

    #should be collecting stats here on the simulation...put the results in a bootstrap,
    #then write the bootstrap to a file
    run = 0
    run_length = (options.learningpres + options.testingpres) * trial_length * options.dt
    logging.info("Run length: %g" % (run_length))

    controller = network._get_node('EC')
    errors = []

    while run < num_runs:
       logging.info("Starting run: %d" % (run + 1))
       network.run(run_length, options.dt)
       network.reset()

       if options.replacecleanup:
           replace_cleanup(network, D, N, threshold_lo, threshold_hi, learning_rate)

       if options.replacevectors:
           controller.generate_vectors()

       logging.info("Done run: %d" % (run + 1))
       run += 1

       errors.append(controller.latest_rmse)
       logging.info("RMSE for run: %g" % (controller.latest_rmse))

    if len(errors) > 0:
        mean = np.mean
        def var(x):
            if len(x) > 1:
                return float(sum((np.array(x) - mean(x))**2)) / float(len(x) - 1)
            else:
                return 0

        ci = bootstrapci(errors, mean, n=999, p=.95)

        config = ConfigParser.ConfigParser()

        config.add_section('Options')
        for attr in dir(options):
            item = getattr(options, attr)
            if not callable(item) and attr[0] != '_':
                config.set('Options', attr, str(item))

        config.add_section('Error')
        config.set('Error', 'raw', str(errors))
        config.set('Error', 'mean', str(mean(errors)))
        config.set('Error', 'var', str(var(errors)))
        config.set('Error', 'lowCI', str(ci[0]))
        config.set('Error', 'hiCI', str(ci[1]))

        f = open(options.resultsfile, 'w')
        config.write(f)
        f.close()


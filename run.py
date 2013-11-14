#run.py
import hrr
import cleanup_lib.cleanup_utilities as cu
import cleanup_lib.learn_cleanup as lc
from stats.bootstrapci import bootstrapci
from ca.nengo.util.impl import NodeThreadPool

from optparse import OptionParser
import sys
import random
import ConfigParser
import numeric as np
import logging

def make_simple_test_schedule(learning, testing):
    def f():
        yield learning
        yield testing
    return f

def simple_noise(noise):

    def f(input_vec):
        v = input_vec + noise * hrr.HRR(D).v
        v = v / np.sqrt(sum(v**2))
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

        reconstruction = S.convolve(~noise_vocab[partner_key])
        reconstruction.normalize()

        return reconstruction.v

    return f

def reduced_run(options):
    radius = options.radius
    options.cleanup_neurons = \
            cu.minimum_neurons(options.Plo, options.Vlo,
                               options.threshold - options.testing_bias, options.D, (100, 5000))

    options.threshold = (options.threshold, options.threshold)
    options.max_rate = (options.max_rate, options.max_rate)

    run(options)

def normal_run(options):
    N = options.cleanup_neurons
    D = options.D

    P_hi = options.Phi
    P_lo = options.Plo

    V_hi = options.Vhi
    V_lo = options.Vlo

    _, threshold_lo = cu.minimum_threshold(P_lo, V_lo, N, D)
    _, threshold_hi = cu.minimum_threshold(P_hi, V_hi, N, D)

    options.radius = threshould_hi
    options.max_rate = (100,200)
    threshold_ratio = float(threshold_lo)/float(threshold_hi)
    options.threshold = (threshold_ratio, threshold_ratio + 0.8 * (1 - threshold_ratio))

    run(options)

def run(options):
    logging.info("Threshold: " + str(options.threshold))
    logging.info("Num cleanup neurons: " + str(options.cleanup_neurons))
    logging.info("Radius: " + str(options.radius))

    if options.noise_type == "hrr":
        options.learning_noise = hrr_noise(options.D, options.learning_noise)
        options.testing_noise = hrr_noise(options.D, options.testing_noise)
    else:
        options.learning_noise = simple_noise(options.learning_noise)
        options.testing_noise = simple_noise(options.testing_noise)

    options.schedule_func = make_simple_test_schedule(options.learningpres, options.testingpres)

    options_dict = options.__dict__

    network = lc.make_learnable_cleanup(**options_dict)

def parse_args():
    parser = OptionParser()

    parser.add_option("-N", "--numneurons", dest="cleanup_neurons", default=None, type="int", 
            help="Number of neurons in cleanup")

    parser.add_option("-n", "--neuronsperdim", dest="neurons_per_dim", default=20, type="int", 
            help="Number of neurons per dimension for other ensembles")

    parser.add_option("-D", "--dim", dest="D", default=16, type="int", 
            help="Dimension of the vectors that the cleanup operates on")

    parser.add_option("-V", "--numvectors", dest="num_vecs", default=4, type="int", 
            help="Number of vectors that the cleanup will try to learn")

    parser.add_option("--dt", default=0.001, type="float", help="Time step")

    parser.add_option("--cleanup-pstc", dest='cleanup_pstc', default=0.02, type="float", 
            help="Time constant for post-synaptic current of cleanup neurons")

    parser.add_option("-a", "--alpha", dest="learning_rate", default=5e-5, type="float", help="Learning rate")

    parser.add_option("-L", "--triallength", dest="trial_length", default=100, type="int", 
            help="Length of each vector presentation, in timesteps")

    parser.add_option("-P", "--learningpres", default=100, type="int", 
            help="Number of presentations during learning")

    parser.add_option("-p", "--testingpres", default=20, type="int", 
            help="Number of presentations during testing")

    parser.add_option("-T", "--learningnoise", dest="learning_noise", default=1, type="float", 
            help="Parameter for the noise during learning")

    parser.add_option("-t", "--testingnoise", dest="testing_noise", default=1, type="float", 
            help="Parameter for the noise during testing")

    parser.add_option("--noise-type", dest="noise_type", default="hrr", 
            help="Type of noise to use")

    parser.add_option("-R", "--numruns", default=0, type="int", 
            help="Number of runs to do. We can reuse certain things between runs, \
                  so this speeds up the process of doing several runs at once")

    parser.add_option("--control-bias", dest="user_control_bias", default=True, 
            help="Whether to use different biases during learning and testing")

    parser.add_option("--control-learning", dest="user_control_learning", default=True, 
            help="Whether user controls learning schedule")

    parser.add_option("-B", "--learningbias", dest="learning_bias", default=0.25, type="float", 
            help="Amount of bias during learning. Only has an effect if varthresh is True")

    parser.add_option("-b", "--testingbias", dest="testing_bias", default=-0.1, type="float", 
            help="Amount of bias during testing. Only has an effect if varthresh is True")

    parser.add_option("--Phi", default=.9, type="float", help="Probability for hi")
    parser.add_option("--Plo", default=.9, type="float", help="Probability for low")

    parser.add_option("--Vhi", default=10, type="int", help="Number of neurons for hi")
    parser.add_option("--Vlo", default=50, type="int", help="Number of neurons for low")

    parser.add_option("--threads", default=8, type="int", 
            help="Number of threads to use to run the simulation")

    parser.add_option("--resultsfile", dest="results_file", default="results", help="Name of file to write results to")
    parser.add_option("--logfile", default="log", help="Name of file to log to")

    parser.add_option("-c", "--cleanlearning", dest="clean_learning", default=False, 
            help="Whether to set decoders using clean vectors (supervised) or not (unsupervised)")

    parser.add_option("-I", "--neuralinput", dest="neural_input", default=False, 
            help="Whether the input should be passed through a neural population first")

    parser.add_option("--replacecleanup", default=True, 
            help="Whether to generate a new cleanup neural population for each run.")

    parser.add_option("--replacevectors", default=True, 
            help="Whether to generate new vocabulary vectors for each run.")

    parser.add_option("--errorlearning", default=False, 
            help="Whether to use error for learning \
                 (alternative is to use the learning vector as is).")

    parser.add_option("--dry-run", dest="dry_run", action='store_true', default=False, 
            help="Whether to exit right away (for testing purposes)")

    parser.add_option("--reduced-mode", dest="reduced_mode", action='store_true', default=True, 
            help="In reduced mode, Vhi, Vlo, Phi, Plo all ignored. \
                    Uses max-firing-rate and radius")

    parser.add_option("--max-rate", dest="max_rate", default=200, type="float", 
            help="Maximum firing rate of neurons")

    parser.add_option("--radius", default=1.0, type="float", 
            help="Range of values neurons sensitive to. Only used in reduced mode.")

    parser.add_option("--threshold", default=0.3, type="float", 
            help="Value for intercept of neural tuning curves. Only used in reduced mode")

    (options, args) = parser.parse_args()
    print "options: ", options
    print "args: ", args
    return (options, args)


def run_batch(network, options, args):
    run = 0
    run_length = (options.learningpres + options.testingpres) * trial_length * options.dt
    logging.info("Run length: %g" % (run_length))

    controller = network._get_node('EC')
    errors = []

    while run < options.num_runs:
       logging.info("Starting run: %d" % (run + 1))
       network.run(run_length, options.dt)
       network.reset()

       if options.replacecleanup:
           replace_cleanup(network, **options.__dict__)

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

        f = open(options.results_file, 'w')
        config.write(f)
        f.close()

def dry_run(results_file):
    logging.info("Dry run!")
    config = ConfigParser.ConfigParser()

    config.add_section('Error')
    config.set('Error', 'mean', 0.0)
    config.set('Error', 'var', 0.0)

    f = open(results_file, 'w')
    config.write(f)
    f.close()
    sys.exit()

def start():    
    (options, args) = parse_args()

    logging.basicConfig(filename=options.logfile, filemode='w', level=logging.INFO)
    logging.info("Parameters: " + str(options) + str(args))

    command_line = 'cl' in args

    if options.dry_run:
        dry_run(options.results_file)

    NodeThreadPool.setNumJavaThreads(options.threads)

    if options.reduced_mode:
        reduced_run(options)
    else:
        normal_run(options)

if __name__=="__main__":
    start()

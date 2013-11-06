import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import subprocess
import datetime
import string
import ConfigParser

results_dir = "/home/e2crawfo/hyperopt/learn_cleanup/"
nengo_location = "/home/e2crawfo/nengo-eb57aaa/"

def make_objective(D, num_neurons, neurons_per_dim, num_vectors,
                    trial_length, learning_pres, testing_pres, clean_learning, learning_noise,
                    testing_noise, num_runs, P_hi, P_lo, V_hi, V_lo, threads):

    static_arg_string = \
            (" cl -D %d -N %d -n %d -V %d -L %g -P %d -p %d -T %g -t %g"
             " -R %d --Phi %g --Plo %g --Vhi %d --Vlo %d --threads %d")

    static_arg_string = \
            static_arg_string % (D, num_neurons, neurons_per_dim,
                                num_vectors, trial_length, learning_pres,
                                testing_pres, learning_noise, testing_noise,
                                num_runs, P_hi, P_lo, V_hi, V_lo, threads)

    def objective(kwargs):
        learn_bias = kwargs['learn_bias']
        test_bias = kwargs['test_bias']
        learning_rate = kwargs['learning_rate']

        date_time_string = str(datetime.datetime.now()).split('.')[0]
        date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":", " ","-"])
        results_file = results_dir + "results_" + date_time_string

        variable_arg_string = (" --varbias True -B %g -b %g"
                               " --alpha %g"
                               " --resultsfile %s")

        variable_arg_string = \
                variable_arg_string % (learn_bias, test_bias, learning_rate, results_file)

        call_string = nengo_location + \
                        "nengo-cl " + \
                        nengo_location + \
                        "learn_cleanup.py " + \
                        static_arg_string + \
                        variable_arg_string
        #call_string = call_string.split(' ')

        subprocess.check_call(call_string, shell=True)

        config = ConfigParser.ConfigParser()
        config.read(results_file)

        mean_error = config.getfloat('Error', 'mean')
        error_variance = config.getfloat('Error', 'var')
        print mean_error
        print error_variance

        return {
            'loss': mean_error,
            'loss_variance': error_variance,
            'status': STATUS_OK,
            }

    return objective


if __name__ == "__main__":

    D = 32
    num_neurons = 600
    neurons_per_dim = 50
    num_vectors = 4
    trial_length = 100 #(timesteps)
    learning_pres = 50
    testing_pres = 20
    clean_learning = False
    learning_noise = 0.5
    testing_noise = 0.5
    num_runs = 5
    P_hi = 0.9
    P_lo = 0.9
    V_hi = 10
    V_lo = 20
    threads = 8

    objective = make_objective(D, num_neurons, neurons_per_dim, num_vectors,
                                trial_length, learning_pres, testing_pres, clean_learning,
                                learning_noise,testing_noise, num_runs, P_hi, P_lo, V_hi, V_lo, threads)

    #trials = Trials()
    trials = MongoTrials()

    space = {
            'learn_bias':hp.uniform('learn_bias', 0.0, 1.0),
            'test_bias':hp.uniform('test_bias', 0.0, 1.0),
            'learning_rate':hp.uniform('learning_rate', 5e-6, 5e-4),
            }

    best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials)

    #print trials.trials
    #print trials.results
    #print trials.losses()
    #print trials.statuses()

print best


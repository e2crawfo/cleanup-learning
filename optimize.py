import pickle
import time
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
import datetime
import string
import subprocess
import multiprocessing
import argparse
import os

def_logging_dir = "/home/ctnuser/e2crawfo/cleanup-learning/logs/"
def_results_dir = "/home/ctnuser/e2crawfo/cleanup-learning/temp/"
def_nengo_path = "/home/ctnuser/Nengo/nengo-1_4/"

def make_objective(nengo_path, results_dir, log_file_prefix, date_time_string, D, num_neurons, neurons_per_dim, num_vectors,
                    trial_length, learning_pres, testing_pres, clean_learning, learning_noise,
                    testing_noise, num_runs, P_hi, P_lo, V_hi, V_lo, threads, dry_run):

    static_arg_string = \
            (" cl -D %d -N %d -n %d -V %d -L %g -P %d -p %d -T %g -t %g"
             " -R %d --Phi %g --Plo %g --Vhi %d --Vlo %d --threads %d")

    if dry_run:
        static_arg_string += " --dry-run"

    static_arg_string = \
            static_arg_string % (D, num_neurons, neurons_per_dim,
                                num_vectors, trial_length, learning_pres,
                                testing_pres, learning_noise, testing_noise,
                                num_runs, P_hi, P_lo, V_hi, V_lo, threads)

    def objective(kwargs):
        #mongo needs these imported here, rather than up top
        import string
        import os
        import subprocess
        import ConfigParser
        from hyperopt import STATUS_OK

        learn_bias = kwargs['learn_bias']
        test_bias = kwargs['test_bias']
        learning_rate = kwargs['learning_rate']

        arg_string = "_B_%g_b_%g_alpha_%g" % (learn_bias, test_bias, learning_rate)
        results_file = results_dir + "results_" + date_time_string + arg_string
        log_file = log_file_prefix + arg_string

        variable_arg_string = (" --varbias True -B %g -b %g"
                               " --alpha %g"
                               " --resultsfile %s")

        variable_arg_string = \
                variable_arg_string % (learn_bias, test_bias, learning_rate, results_file)

        call_string = nengo_path + \
                        "nengo-cl " + \
                        os.getcwd() + \
                        "/learn_cleanup.py " + \
                        static_arg_string + \
                        variable_arg_string + \
                        " --logfile " + \
                        log_file

        subprocess.check_call(call_string, shell=True)

        config = ConfigParser.ConfigParser()
        config.read(results_file)

        mean_error = config.getfloat('Error', 'mean')
        error_variance = config.getfloat('Error', 'var')
        print "Mean:", mean_error
        print "Variance", error_variance

        return {
            'loss': mean_error,
            'loss_variance': error_variance,
            'status': STATUS_OK,
            }

    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn a cleanup memory.')

    parser.add_argument('--nengo',
            default=def_nengo_path,
            type=str,
            help='Full path to nengo 1.4')
    parser.add_argument('--mongo',
            action='store_true',
            help='Whether to use mongo to evaluate points in parallel')
    parser.add_argument('--mongo-workers',
            dest='mongo_workers',
            default=2,
            type=int,
            help='Number of parallel workers to use to evaluate points.'\
                  ' Only has an effect if --mongo is also supplied')
    parser.add_argument('--exp-key',
            dest='exp_key',
            default='exp1',
            type=str,
            help='Unique key identifying this experiment within the mongodb')
    parser.add_argument('--dry-run',
            dest='dry_run',
            default=False,
            action='store_true',
            help='A dry run will not evaluate the function. Useful for testing the'\
            'hyperopt framework without having to wait for the function evaluation')


    argvals = parser.parse_args()

    nengo_path = argvals.nengo
    use_mongo = argvals.mongo
    num_mongo_workers = max(argvals.mongo_workers, 1)
    exp_key = argvals.exp_key
    dry_run = argvals.dry_run

    D = 16
    num_neurons = 500
    neurons_per_dim = 50
    num_vectors = 4
    trial_length = 100 # of timesteps
    learning_pres = 1
    testing_pres = 2
    clean_learning = False
    learning_noise = 0.5
    testing_noise = 0.5
    num_runs = 3
    P_hi = 0.9
    P_lo = 0.9
    V_hi = 10
    V_lo = 20
    threads = 8

    date_time_string = str(datetime.datetime.now()).split('.')[0]
    date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":", " ","-"])
    log_file_prefix = def_logging_dir + "temp_" + date_time_string

    objective = make_objective(nengo_path, def_results_dir, log_file_prefix, date_time_string, D, num_neurons, neurons_per_dim,
                                num_vectors, trial_length, learning_pres, testing_pres, clean_learning,
                                learning_noise,testing_noise, num_runs, P_hi, P_lo, V_hi, V_lo, threads, dry_run)

    if use_mongo:
        trials = MongoTrials('mongo://localhost:1234/first_try/jobs', exp_key=exp_key)

        worker_call_string = \
            ["hyperopt-mongo-worker",
            "--mongo=localhost:1234/first_try",
            "--max-consecutive-failures","1",
            "--reserve-timeout", "15.0",
            "--workdir",def_results_dir,
            ]

        print worker_call_string
        workers = []
        for i in range(num_mongo_workers):
            #using Popen causes the processes to run in the background
            p = subprocess.Popen(worker_call_string)
            workers.append(p)
    else:
        trials = Trials()

    space = {
            'learn_bias':hp.uniform('learn_bias', -.1, .5),
            'test_bias':hp.uniform('test_bias', -.1, .5),
            'learning_rate':hp.uniform('learning_rate', 5e-6, 5e-4),
            }

    
    then = datetime.datetime.now()

    best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals=5,
        trials=trials)

    now = datetime.datetime.now()

    if use_mongo:
        for p in workers:
            p.terminate()

        #merge the temporary log files
        filenames = os.listdir(def_logging_dir)
        temp_logs = filter(lambda x: x.find("temp_" + date_time_string) > -1, filenames)

        aggregated_log = open(def_logging_dir + "log_" + date_time_string, 'w')
        
        for temp_log in temp_logs:
            t = open(def_logging_dir + temp_log, 'r')
            aggregated_log.write(t.read())
            t.close()
            os.remove(def_logging_dir + temp_log)
        
        aggregated_log.write("Time for fmin: " + str(now - then) + "\n")
        aggregated_log.write("Trials: " + str(trials.trials) + "\n")
        aggregated_log.write("Results: " + str(trials.results) + "\n")
        aggregated_log.write("Losses: " + str(trials.losses()) + "\n")
        aggregated_log.write("Statuses: " + str(trials.statuses()) + "\n")
        
        aggregated_log.close()

print best


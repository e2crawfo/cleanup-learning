import datetime

from hyperopt import plotting as hplot
from hyperopt.mongoexp import MongoTrials
from bson.son import SON
from bson.objectid import ObjectId

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def read_log_file(filename):
    f = open(filename, 'r')

    line = ''

    while line.find('Trials:') < 0:
        line = f.readline()
    
    trials = line
    results = f.readline()
    losses = f.readline()
    statuses = f.readline()
    f.close()

    trials = trials.split('Trials: ')[1]
    exec "trials=" + trials
    return trials

if __name__=="__main__":
    trials = MongoTrials('mongo://localhost:1234/first_try/jobs', exp_key='big_run3')

    x_name = 'test_bias'
    y_name = 'learn_bias'
    z_name = 'learning_rate'

    x = trials.vals[x_name]
    y = trials.vals[y_name]
    z = trials.vals[z_name]
    w = trials.losses()

    indices = filter(lambda x: w[x] < 1, range(len(w)))
    x = [x[i] for i in indices]
    y = [y[i] for i in indices]
    z = [z[i] for i in indices]
    w = [w[i] for i in indices]

    p = ax.scatter(x, y, z, cmap='winter', s=50, c=w)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    fig.colorbar(p)
    plt.show()

    sorted_indices = sorted(range(len(w)), key=lambda x: w[x])
    print w
    print sorted_indices

    num = 3
    print "Best ", num
    for j in range(num):
        i = sorted_indices[j]
        print "Loss: ", w[i], ", Index: ", i, ", ", x_name, ": ", x[i], \
                ", ", y_name, ": ", y[i], ", ", z_name, ": ", z[i]

    #plt.subplot(311)
    #hplot.main_plot_history(trials, do_show=False)
    #plt.subplot(312)
    #hplot.main_plot_histogram(trials, do_show=False)
    #plt.subplot(313)
    #hplot.main_plot_vars(trials, do_show=True)
    
    

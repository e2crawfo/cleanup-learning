import numpy as np
import matplotlib.pyplot as plt
from oja_timing import run
import seaborn as sns
sns.set(font='Droid Serif')
from mytools import hrr, nf, fh, nengo_stack_plot
import itertools

#dims = [(512, 126), (512, 64), (512, 32), (256, 64), (256, 32), (256, 16)]
#cleanup_n = [10,50,100,250,500,1000]

#points = itertools.product([16, 32, 64], [4, 8, 16])
points = itertools.product([16, 16], [4])
points = filter(lambda x: x[0] > x[1] and x[0] % x[1] == 0, points)
print points
#dims = [(0,p[0], p[1]) for p in points]
dims = [(1, p[0], p[1]) for p in points]

line_styles = {0: '-', 1:'--'}

cleanup_n = [100, 200, 300, 400, 500]

dim_data = {key: [] for key in dims}

for d in dims:
    i = 0
    for cn in cleanup_n:
        t1, t2 = run(d[0], d[1], d[2], cn)
        dim_data[d].append(t1)
        i += 1

data = np.zeros((0, len(cleanup_n)))
for key in dims:
    plt.plot(cleanup_n, dim_data[key], ls=line_styles[key[0]], label=str(key))

plt.legend()

file_config = {}

filename = fh.make_filename('runtimes', directory='runtimes',
                            config_dict=file_config, extension='.png')
plt.savefig(filename)

plt.show()


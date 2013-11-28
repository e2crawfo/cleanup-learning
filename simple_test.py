import hrr
import nef
import random

D = 512

N_items = 1
N_per_item = 20
N_per_D = 50

neural_input = False
neural_output = False

def thresh(x):
    if x > 0.2:
        return x
    else:
        return 0.0

def id(x):
    return x

def hrr_noise(D, num, noise_vocab=None):

    if not noise_vocab:
        noise_vocab = hrr.Vocabulary(D)

    def f(input_vec):
        keys = [noise_vocab.parse('x' + str(x)) for x in range(2*num+1)]

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

max_noise = 10
v = hrr.Vocabulary(D)
vector = v.parse("A").v

noise_dict = {}
t = 0.0
dt = 0.1
for i in range(1, max_noise+1):
    noise_func = hrr_noise(D, i, v)
    noise_dict[t] = noise_func(vector)
    t+=dt

print noise_dict
#def input_func(t):
#    global noise_vector, time_index, noise_index
#
#    print t
#    if time_index == -1 or (time_index == trial_length and noise_index<max_noise):
#        noise_index += 1
#        time_index = 0
#        noise_func = hrr_noise(D, noise_index, v)
#        noise_vector = noise_func(vector)
#    else:
#        time_index += 1
#
#    return list(noise_vector)

radius = 1.0
threshold = (0.0, 0.2)
max_rate=(200,200)

net = nef.Network('simple_test', seed=2)

#input = noise_func(vector.v)
#input=vector.v

net.make_input('in', values=noise_dict)

if neural_input:
    net.make_array('input', neurons=N_per_D, length=D, dimensions=1, mode='default')
else:
    net.make('input', neurons=1, dimensions=D, mode='direct')

net.connect('in', 'input', pstc=0.001)

if neural_input:
    net.make_array('output', neurons=N_per_D, length=D, dimensions=1, mode='default')
else:
    net.make('output', neurons=1, dimensions=D, mode='direct')

for vec_str in v.hrr:
    if isinstance(vec_str, type('')):
        name = 'cleanup_' + vec_str
        vec = v.hrr[vec_str].v
        net.make(name, neurons=N_per_item, dimensions=1, radius=radius, intercept=threshold, max_rate=max_rate, tau_ref=0.002, encoders=[[1]] * N_per_item)
        net.connect('input', name, transform=vec, pstc=0.001)
        net.connect(name, 'output', func=thresh, transform=vec, pstc=0.001)



net.add_to_nengo()
net.view()










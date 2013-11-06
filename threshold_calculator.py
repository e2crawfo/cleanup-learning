__author__ = 'e2crawfo'

#from sys import float_info
import numeric as np
#import matplotlib.pyplot as plt
import hrr as hrr
import operator
from org.python.core.util.ExtraMath import EPSILON

"""
This accepts a probability value, P, and a number of neurons, v, and draws a graph, of
neural threshold vs number of neurons in the cleanup population such that there is a
probability of P that an arbitrary fixed vector is close enough to at least v neurons to activate
them
"""

def binary_search_fun(func, item, bounds=(0,1), non_inc=False, eps=0.00001):
  """binary search a monotonic function

  :param func: monotonic function defined on the interval "bounds"
  :type func: function

  :param item: the value we want to match. we are trying to find a float c st f(c) ~= item
  :type item: float

  :param non_inc: use to specify if func is non-increasing rather than non-decreasing. if its True,
                  we just flip both func and the item in the x-axis and proceed as normal
  :type non_inc: boolean

  :param bounds: the bounds in the domain of f
  :type bounds: float

  :param eps: the threshold for terminating. the value of func at the domain element return by
              this function will be at least within eps of item
  :type eps: float
  """

  if non_inc:
    f = lambda x: -func(x)
    item = -item
  else:
    f = func

  if item < f(bounds[0]):
    print "Stopping early in bshf, no item can exist"
    return None
  if item > f(bounds[1]):
    print "Stopping early in bshf, no item can exist"
    return None

  return bsfh(f, item, bounds[0], bounds[1], eps)


def bsfh(f, item, lo, hi, eps=EPSILON):

  while not float_eq(lo, hi):
    c = float(lo+hi) / 2.0
    f_c = f(c)
    if float_eq(item, f_c, eps):
      return c
    elif item < f_c:
      hi = c
    else:
      lo = c

  return None

def float_eq(a, b, eps=EPSILON):
  return abs(a - b) <= eps

def binomial(n, i, p, bc):
  return bc(n, i) * p**i * (1 - p)**(n-i)

def binomial_sum(n, p, lo, hi, bcf):
  return np.sum(np.array( [binomial(n, i, p, bcf) for i in range(lo,hi+1)] ) )

def check_monotonic(f, op, step=0.001, bounds=(0,1)):
  num_steps = int(float(bounds[1] - bounds[0]) / float(step))
  pair_gen = ((f(bounds[0] + i * step), f(bounds[0] + (i + 1) * step)) for i in xrange(num_steps))
  return all(op(pair[0], pair[1]) for pair in pair_gen)

def minimum_threshold(P, V, N, D, use_normal_approx=False):
    """
    Return the minimum threshold required to have a probability of at least P that at least V
    neurons are activated by a randomly chosen vector (a vector to be learned) given a population
    of N neurons and vectors of dimension D
    """
    vocab = hrr.Vocabulary(D)

    binomial_coefficients=[]
    bc = 1
    for i in range(0, V):
      binomial_coefficients.append(bc)
      bc *= float(N - i) / float(i + 1)

    binomial_coefficients = np.array(binomial_coefficients)

    bcf = lambda n, i: binomial_coefficients[i]

    #if use_normal_approx:
    #  f = lambda p: 1 - sp.stats.norm(N*p, N * p * (1-p)).cdf
    #else:
    f = lambda p: 1 - binomial_sum(N, p, 0, V-1, bcf)

    prob = binary_search_fun(f, P, eps=0.0001)

    g = lambda t: vocab.prob_within_angle(np.arccos(t))
    threshold = binary_search_fun(g, prob, non_inc=True, eps=0.0001)

    return prob, threshold

def gen_data(D, P, v, N, use_normal_approx=False):
  """
  Together, P, v, and any n element of N define an inequality of the form
  sum(i = v to N)[(n choose i) * p**i * (1 - p) ** (n-i)] >= P. Our goal is to solve for p.
  p and D together determine the required threshold.

  :param P: probability value
  :type P: float

  :param v: number of neurons
  :type v: integer

  :param N: cleanup population sizes to evaulate the required threshold at
  :type N: list of integers
  """


  thresholds = []
  probs = []
  for n in N:
    prob, threshold = minimum_threshold(P, v, n, D, use_normal_approx)

    probs.append(prob)
    thresholds.append(threshold)

    print "n:, ", n, "prob: ", prob, "thresh: ", threshold

  return probs, thresholds

def plot_data(P, v, N, probs, thresholds):
  plt.subplot(211)
  plt.plot(N, probs, marker='o')
  plt.xlabel("Number of Cleanup Neurons")
  plt.ylabel("Probability for Binomial")
  plt.title("Probability for binomial vs Number of Cleanup Neurons, P = %f, v = %d" % (P,v))
  plt.subplot(212)
  plt.plot(N, thresholds, marker='o')
  plt.xlabel("Number of Cleanup Neurons")
  plt.ylabel("Threshold")
  plt.title("Required Threshold vs Number of Cleanup Neurons, P = %f, v = %d" % (P,v))
  plt.show()

if __name__ == "__main__":
  D = 512
  N = [2000 * (i + 1) for i in range(25)]
  P = 0.5
  v = 20

  p, t = gen_data(D, P, v, N)
  plot_data(P, v, N, p, t)


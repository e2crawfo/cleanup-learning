__author__ = 'e2crawfo'

try:
    import numpy as np
except:
    import numeric as np

try:
    import sys
    EPSILON = sys.float_info.epsilon
except:
    from org.python.core.util.ExtraMath import EPSILON

plotting = True
try:
    import matplotlib.pyplot as plt
except:
    plotting = False

import operator

"""
This accepts a probability value, P, and a number of neurons, v, and draws a graph, of
neural threshold vs number of neurons in the cleanup population such that there is a
probability of P that an arbitrary fixed vector is close enough to at least v neurons to activate
them
"""

class ProbabilityCalculator(object):

    prob_normal=None
    last_val=None

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def prob_angle(self, angle):
        """
        See http://yamlb.wordpress.com/2008/05/20/why-2-random-vectors-are-orthogonal-in-high-dimention/
        for argument that the probability of two random vectors being a given angle apart is
        proportional to sin(angle)^(D-2)
        """
        return np.sin(angle)**(self.dimensions-2)

    def prob_within_angle(self, angle, steps=10000, use_cache=True):
        """
        Calculates the probability that the angle between two random vectors is less than angle. Essentially
        integrates the function prob_angle with respect to angle. Has a caching system, so that previous results
        can be reused.
        """
        #steps is actually steps per some angle

        if self.prob_normal is not None:
          denom = self.prob_normal
        else:
          denom=0
          ddenom=1/float(steps)

          for i in range(int(steps * np.pi)):
            denom += self.prob_angle(ddenom*i)

          denom*=ddenom
          self.prob_normal = denom

        #format of last val is (angle, value)
        dnum= 1/float(steps)
        if self.last_val is not None and use_cache:

          if self.last_val[0] < angle:
            start, end = angle, self.last_val[0]
          else:
            start, end = self.last_val[0], angle

          remaining_angle = start - end

          added_num = 0
          for i in range(int(steps * remaining_angle)):
            added_num += self.prob_angle(np.pi-start+dnum*i)
          added_num *= dnum

          if self.last_val[0] < angle:
            num = self.last_val[1] + added_num
          else:
            num = self.last_val[1] - added_num

          self.last_val = (angle, num)

        else:
          num=0
          for i in range(int(steps * angle)):
            num += self.prob_angle(np.pi-angle+dnum*i)
          num*=dnum
          self.last_val = (angle, num)

        prob_within=num/denom

        return prob_within

    def prob_cleanup(self, similarity, vocab_size):
        """
        Calculates the probability of building a "perfect" cleanup, given some fixed number of vectors that
        that it should store and fixed maximum similarity between stored vectors. Basically, if two of the encoding vectors in the cleanup
        (which are assumed to be chosen randomly) are more similar than the angle corresponding to similarity, then this is
        considered a failure, because the cleanup will not be able to accurately separate inputs corresponding to these two
        stored vectors. The probability of building a perfect cleanup then is the probability that no two vectors are that
        similar. Not yet completely sure that this independence assumption is warranted.
        """
        angle=np.arccos(similarity)
        perror = self.prob_within_angle(angle)
        pcorrect=(1-perror)**(vocab_size * (vocab_size + 1) / 2)
        return pcorrect


def binary_search_fun(func, item, bounds=(0,1), non_inc=False, eps=0.00001, verbose=False, integers=False):
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
  f_lo = f(bounds[0])
  f_hi = f(bounds[1])

  if item < f_lo:
    print "Stopping early in bshf, no item can exist. f(lo) = %g", f_lo
    return None
  if item > f_hi:
    print "Stopping early in bshf, no item can exist. f(hi) = %g", f_hi
    return None

  return bsfh(f, item, bounds[0], bounds[1], eps, verbose, integers)


def bsfh(f, item, lo, hi, eps=EPSILON, verbose=False, integers=False):

  while not float_eq(lo, hi):

    c = float(lo+hi) / 2.0
    if integers:
        c = int(c)

    f_c = f(c)
    if verbose:
        print "Trying point %g, f(%g) = %g, lo: %g, hi: %g" % (c, c, f_c, lo, hi)

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
  terms = np.array( [binomial(n, i, p, bcf) for i in range(lo,hi+1)] ) 
  result = np.sum(terms)
  return result

factorial_dict={}
def factorial(m):

    f = 1
    n = m

    while n > 1:
        if n in factorial_dict:
            f = f * factorial_dict[n]
            break

        f = f * n
        n-=1

    factorial_dict[m] = f
    return f
        
def choose(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))

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

    prob_calc = ProbabilityCalculator(D)

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

    g = lambda t: prob_calc.prob_within_angle(np.arccos(t))
    threshold = binary_search_fun(g, prob, non_inc=True, eps=0.0001)

    return prob, threshold


def minimum_neurons(P, V, T, D, bounds, use_normal_approx=False, verbose=False):
    """
    Return the minimum number of neurons required to have a probability of at least P that at least V
    neurons are activated by a randomly chosen vector (a vector to be learned) given neurons with 
    threshold T and vectors of dimension D
    """

    prob_calc = ProbabilityCalculator(D)
    p = prob_calc.prob_within_angle(np.arccos(T))

    f = lambda N: 1 - binomial_sum(N, p, 0, V-1, choose)

    neurons = binary_search_fun(f, P, bounds=bounds, eps=0.01, verbose=verbose, integers=True)
    return neurons

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
  if plotting:
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
    minimum_neurons(0.9, 10, .5, 16, bounds=(100,1000), verbose=True)
#  D = 512
#  N = [2000 * (i + 1) for i in range(25)]
#  P = 0.5
#  v = 20
#
#  p, t = gen_data(D, P, v, N)
#  plot_data(P, v, N, p, t)
#


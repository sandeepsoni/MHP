from __future__ import division
import numpy as np

def kernel (x,y,b):
	""" calculates the exponentially decay kernel for difference of x-y with bandwidth b"""
	return np.exp (-b * (x-y))

def drawExpRV (param, rng):
	# using the inverse method
	#return (1/float (param)) * np.log (rng.uniform (0,1))

	# using the built-in numpy function
	return rng.exponential (scale=param)

def attribute (uniform_rv, iStar, mlambda):
	# this is the recommended
	S = np.cumsum (mlambda) / iStar
	return (uniform_rv > S).sum()	

def attribute2 (uniform_rv, intensities):
    # The more traditional way
	S = np.cumsum (intensities) / np.sum (intensities)
	return (uniform_rv > S).sum ()

def attribute3 (intensities):
	# the fast way using numpy
	p = intensities / np.sum (intensities)
	return np.random.choice(len(intensities), 1, p=p)	


def HPIntensities (mu, alpha, omega, history, t):
	""" returns the intensities for all dimensions as a vector at time t"""
	nEvents = len (history)
	I = np.copy(mu)
	for i in xrange (nEvents):
		src, ts = history[i]
		I = I + alpha[src, :] * kernel (t,ts,omega[src,:])

	return np.squeeze (I)

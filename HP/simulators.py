"""
This module contains Hawkes Process simulators.
HP is a stochastic generative process for events in continuous time
"""
from __future__ import division
import sklearn
import numpy as np
import HPUtils

def univariate (mu, alpha, omega, T, numEvents=None, seed=None):
	""" generates events based on univariate hawkes process
		with an exponential decay kernel
	

	Parameters:
	-----------
	mu: float
		The base intensity.

	alpha: float
		The self-excitation.

	omega: float
		The bandwidth parameter.
		
	T: float
		The final time. All event times lie within the interval [0,T]

	numEvents: int
		The number of events to be generated. The default is None.
		If None, events are generated upto time T.
		If not None, attempt to generate upto `numEvents` events
		under the condition that the time does not exceeed T.
		
	seed: int
		The seed for the random number generation. Defaults to None. 
		This ensures repeatability in the generation process.
		For debugging one should specify a constant seed. 
		But otherwise the seed should be `None`.

	Returns:
	--------

	history: list
		The timestamps of the process.

	Notes:
	------

	Refer to the simulation algorithm given in the slides:
	http://lamp.ecp.fr/MAS/fiQuant/ioane_files/HawkesCourseSlides.pdf


	This code is also adapted heavily or inspired from the Hawkes R package.
	https://cran.r-project.org/web/packages/hawkes/

	and
	https://github.com/stmorse/hawkes/
	"""

	nTotal = 0
	if numEvents is None:
		nExpected = np.iinfo(np.int32).max
	else:
		nExpected = numEvents

	# Initialization
	prng = sklearn.utils.check_random_state (seed)
	lambda_star = mu
	dlambda = 0.0
	t,s = 0,0

	history = list ()

	# In the absence of history, the first timestamp is generated
	# from an exponential distribution.
	s = s + HPUtils.drawExpRV (lambda_star, prng)
	
	if s <= T and nTotal < nExpected:
		history.append (s)
		dlambda = alpha
		t=s
		nTotal += 1
	else:
		return history

	# General routine.
	while nTotal < nExpected:
		lambda_star = mu + dlambda * HPUtils.kernel (s, t, omega)
		s = s + HPUtils.drawExpRV (lambda_star, prng)
		if s > T:
			break

		d = prng.uniform (0,1)
		# accept only if the following condition holds
		if d <= (mu + dlambda * HPUtils.kernel (s, t, omega) / lambda_star):
			history.append (s)
			dlambda = dlambda * HPUtils.kernel (s, t, omega) + alpha
			t=s
			nTotal += 1
			
	return history

def multivariate (mu, alpha, omega, T, numEvents=None, checkStability=False, seed=None):
	""" I found out that the simulator that I wrote wasn't completely correct.
	It had the following problems:

	a) Model parameters could not be recovered: Given a cascade that was generated 
	   using the thinning algorithm, one should be able to recover the model parameters
	   by MLE. Unfortunately, this was not happening.
	b) I think my implementation overestimates the number of events per unit time.

	I haven't found out the reasons why my implementation had these problems, but
	I found an implementation from https://github.com/stmorse/hawkes

	and I slightly modified it.

	Parameters:
	-----------
	mu: numpy.array
		The base intensity vector of size K*1

	alpha: scipy.sparse matrix
		The excitation matrix of size K*K
		TODO: change the full dense matrix to sparse.

	omega: float
		The bandwidth parameter.
		
	T: float
		The final time. All the times lie within the interval [0,T]

	numEvents: int
		The number of events to be generated. The default is None.
		If None, events are generated upto time T.
		If not None, attempt to generate upto `numEvents` events
		under the condition that the time does not exceeed T.

	checkStability: boolean
		Before simulating, should we check the stability of HP by spectral analysis.
		
	seed: int
		The seed for the random number generation. Defaults to None. 
		This ensures repeatability in the generation process.
		For debugging one should specify a constant seed. 
		But otherwise the seed should be `None`.
	
	Returns:
	--------

	history: list of tuples
		Every element of the list is a pair: user_id and timestamp of the generated event.

	"""
	prng = sklearn.utils.check_random_state (seed)
	dim = mu.shape[0]
	alpha = alpha.tocsr().toarray() # just keep them dense (I'm sure the simulation can be made faster by exploiting sparsity)
	nTotal = 0
	history = list ()
	# Initialization
	if numEvents is None:
		nExpected = np.iinfo (np.int32).max
	else:
		nExpected = numEvents
	s = 0.0

	if checkStability:
		w,v = np.linalg.eig (alpha)
		maxEig = np.amax (np.abs(w))
		if maxEig >= 1:
			print "(WARNING) Unstable ... max eigen value is: {0}".format (maxEig)

	Istar = np.sum(mu)
	s += HPUtils.drawExpRV (1./Istar, prng)

	if s <=T and nTotal < nExpected:
		# attribute (weighted random sample, since sum(mu)==Istar)
		n0 = int(prng.choice(np.arange(dim), 1, p=(mu / Istar)))
		history.append((n0, s))
		nTotal += 1

	# value of \lambda(t_k) where k is most recent event
	# starts with just the base rate
	lastrates = mu.copy()

	decIstar = False
	while nTotal < nExpected:
		uj, tj = int (history[-1][0]), history[-1][1]

		if decIstar:
			# if last event was rejected, decrease Istar
			Istar = np.sum(rates)
			decIstar = False
		else:
			# otherwise, we just had an event, so recalc Istar (inclusive of last event)
			Istar = np.sum(lastrates) + alpha[uj,:].sum()
			
		s += HPUtils.drawExpRV (1./Istar, prng)
		if s > T:
			break

		# calc rates at time s (use trick to take advantage of rates at last event)
		rates = mu + HPUtils.kernel (s,tj,omega) * (alpha[uj,:] + lastrates - mu)

		# attribution/rejection test
		# handle attribution and thinning in one step as weighted random sample
		diff = Istar - np.sum(rates)
		n0 = int (prng.choice(np.arange(dim+1), 1, p=(np.append(rates, diff) / Istar)))

		if n0 < dim:
			history.append((n0, s))
			# update lastrates
			lastrates = rates.copy()
			nTotal += 1
		else:
			decIstar = True

	return history

def oldMultivariate (mu, alpha, omega, T, numEvents=None, seed=None):
	""" generates events based on multivariate hawkes process
		with an exponential decay kernel.
	

	Parameters:
	-----------
	mu: numpy.array
		The base intensity vector of size K*1

	alpha: scipy.sparse matrix
		The excitation matrix of size K*K
		TODO: change the full dense matrix to sparse.

	omega: numpy.array
		The bandwidth matrix.
		TODO: change the full dense matrix to sparse
		
	T: float
		The final time. All the times lie within the interval [0,T]

	numEvents: int
		The number of events to be generated. The default is None.
		If None, events are generated upto time T.
		If not None, attempt to generate upto `numEvents` events
		under the condition that the time does not exceeed T.
		
	seed: int
		The seed for the random number generation. Defaults to None. 
		This ensures repeatability in the generation process.
		For debugging one should specify a constant seed. 
		But otherwise the seed should be `None`.

	Returns:
	--------

	history: list of tuples
		Every element of the list is a pair: user_id and timestamp of the generated event.

	Notes:
	------

	Refer to the simulation algorithm given in the slides:
	http://lamp.ecp.fr/MAS/fiQuant/ioane_files/HawkesCourseSlides.pdf


	This code is also adapted heavily from the Hawkes R package.
	https://cran.r-project.org/web/packages/hawkes/
	"""

	prng = sklearn.utils.check_random_state (seed)
	dlambda = 0 * alpha.tocsr ()
	mlambda = np.zeros_like (mu)

	nTotal = 0
	t,s = 0,0
	# Initialization
	if numEvents is None:
		nExpected = np.iinfo (np.int32).max
	else:
		nExpected = numEvents

	dimensions = len (mu)
	history = list ()

	lambda_star = np.sum(mu)
	mlambda = np.copy (mu)

	s = s + HPUtils.drawExpRV (lambda_star, prng)

	# first event
	if s <= T and nTotal < nExpected:
		k = HPUtils.attribute (prng.uniform(0,1), lambda_star, mlambda)
		history.append ((k,s))
		#if (len(history) % 100) == 1: print len(history), k, s
		dlambda[k, :] = alpha[k, :] # event by k excites others i.e k -> *
		mlambda = mu + alpha[k,:]
		nTotal += 1
	else:
		return history	

	t=s
	# general routine
	lambda_star = np.sum (mlambda)
	while nTotal < nExpected:
		s = s + HPUtils.drawExpRV (lambda_star, prng)
		if s > T:
			break
		if s <= T:
			d = prng.uniform (0, 1)
			"""# unrolled loop for the following three lines which are vectorized
			im = 0.0
			for i in xrange (dimensions):
				dl = 0.0
				for j in xrange (dimensions):
					dl += dlambda[j, i] * HPSimUtils.kernel (s, t, omega[j,i])
				mlambda[i] = mu[i] + dl
				im += mlambda[i]
			"""
			sumDlambda = (HPUtils.kernel (s,t,omega) * dlambda).sum(axis=0)
			mlambda = mu + np.squeeze (np.asarray(sumDlambda))
			im = np.sum (mlambda)
			
			if d <= (im / lambda_star):
				k = HPUtils.attribute (d, lambda_star, mlambda)
				history.append ((k,s))
				#if (len(history) % 100) == 1: print len(history), k, s
				"""# unrolled loop for the following four lines which are vectorized 
				lambda_star = 0.0
				for i in xrange (dimensions):
					dl = 0.0
					for j in xrange (dimensions):
						dlambda[j,i] = dlambda[j,i] * HPSimUtils.kernel (s,t,omega[j,i])
						if k == j:
							dlambda[k,i] += alpha[k,i]
						dl += dlambda[j,i]
					lambda_star+= mu[i] + dl
				"""
				dlambda = (HPUtils.kernel (s, t, omega) * dlambda)
				dlambda[k,:] += alpha[k,:]
				sumDlambda = np.squeeze (np.asarray(dlambda.sum(axis=0)))
				lambda_star = np.sum (mu + sumDlambda)
				t=s
				nTotal += 1
			else:
				lambda_star = im	

	return history	

def slowMultivariate (mu, alpha, omega, T, numEvents=None, seed=None):
	""" generates events based on multivariate hawkes process
		with an exponential decay kernel.

		This a slower implementation compared to the above which is 
		optimized to be linear in the number of events. In contrast,
		this implementation is quadratic in the number of events, but
		is much easier to follow.
	

	Parameters:
	-----------
	mu: numpy.array
		The base intensity vector of size K*1

	alpha: numpy.array
		The excitation matrix of size K*K
		TODO: change the full dense matrix to sparse.

	omega: numpy.array
		The bandwidth matrix.
		TODO: change the full dense matrix to sparse
		
	T: float
		The final time. All the times lie within the interval [0,T]

	numEvents: int
		The number of events to be generated. The default is None.
		If None, events are generated upto time T.
		If not None, attempt to generate upto `numEvents` events
		under the condition that the time does not exceeed T.
		
	seed: int
		The seed for the random number generation. Defaults to None. 
		This ensures repeatability in the generation process.
		For debugging one should specify a constant seed. 
		But otherwise the seed should be `None`.

	Returns:
	--------

	history: list of tuples
		Every element of the list is a pair: user_id and timestamp of the generated event.

	Notes:
	------

	Refer to the simulation algorithm given in the slides:
	http://lamp.ecp.fr/MAS/fiQuant/ioane_files/HawkesCourseSlides.pdf


	This code is heavily adapted from .
	http://www.cc.gatech.edu/~mfarajta/resources/activity-shaping-code.zip
	"""
	prng = sklearn.utils.check_random_state (seed)

	lambda_star = 0.0
	nTotal = 0
	t,s = 0,0

	# Initialization
	if numEvents is None:
		nExpected = np.iinfo (np.int32).max
	else:
		nExpected = numEvents

	dimensions = len (mu)
	history = list ()

	while nTotal < nExpected:
		maxCumIntensities = np.sum (HPUtils.HPIntensities (mu, alpha, omega, history, t))
		s = s + HPUtils.drawExpRV (maxCumIntensities, prng)
		if s > T:
			break
		instIntensities = HPUtils.HPIntensities (mu, alpha, omega, history, s)
		cumInstIntensities = np.sum (instIntensities)
		d = prng.uniform()
		if d < (cumInstIntensities / maxCumIntensities):
			# accept the point and assign it to some dimension
			u = HPUtils.attribute (d, cumInstIntensities, instIntensities)
			history.append ((u,s))
			nTotal += 1
		t = s

	return history

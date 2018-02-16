"""
This module contains Hawkes Process estimators.
HP is a stochastic generative process for events in continuous time.
"""
from __future__ import division
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
import cvxpy as cvx
import HPUtils

def univariateHLL(params, omega, seq, sign=1.0, verbose=False):
	""" (Negative) Log-likelihood """
	mu, alpha = params[0], params[1]
	last = seq[-1]
	constant_survival = last * mu # survival function dependent upon base intensity
	updating_survival = 0.0
	occurrence_likelihood = 0.0
	R = 0.0
	epsilon = 1e-50
	prev = -np.inf
	for curr in seq:
		R = HPUtils.kernel (curr, prev, omega) * R
		updating_survival += (1 - HPUtils.kernel (last, curr, omega))
		occurrence_likelihood += np.log (mu + alpha * R + epsilon)
		R += 1
		prev = curr

	ll = (sign) * (-constant_survival - alpha*updating_survival + occurrence_likelihood)
	return ll
	
def univariate (seq, omega, verbose=False):
	params = np.random.uniform (0,1,size=2)
	bounds = [(0, None), (0, None)]
	res = minimize(univariateHLL, params, args=(omega, seq, -1.0, verbose), 
				   method="L-BFGS-B",
				   bounds=bounds,
				   options={"ftol": 1e-10, "maxls": 50, "maxcor":50, "maxiter":1000, "maxfun": 1000})

	return res.x[0], res.x[1]

def calculateR (seq, dim, omega):
	""" This function calculates the intermediate quantity which is the 
		unweighted cumulative intensity of past events. This is a naive implementation, 
		where R is allocated to contain dim * dim * nEvents entries (plus extra dim * dim
		to make the recursive update possible). In reality though, it is very sparse.

		Recursively, the equations are as follows:
		R[i][j,t] is defined as the excitation at any given time.
		i is the node that recieves the excitation
		j is the node that has had past events
		t is the time at which we calculate the excitation


		R[i][j,t] = K(t-lasttime(i)) * R[i][j,lasttime(i)] ... i=j
				  = K(t-lasttime(i)) * R[i][j,lasttime(i)] + \sum_{e:lasttime(i) < t_e < t} K(t-t_e) ... i!=j

	"""
	# initialization
	#R = {i: np.zeros ((dim, len (seq) + 1)) for i in xrange (dim)}
	R = {i: sp.lil_matrix((dim, len (seq) + 1)) for i in xrange (dim)}

	#for i in xrange (dim): # this loop is not really required?
		#R[i][:,0] = 0

	lastseen = {i: (-np.inf, 0) for i in xrange (dim)}

	for i, event in enumerate (seq):
		src, curr = int (event[0]), event[1]
		R[src][:,i+1] = HPUtils.kernel (curr, lastseen[src][0], omega) * R[src][:, lastseen[src][1] + 1]
		# do per node differently
		subseq = seq[lastseen[src][1]:i]
		for e in subseq:
			n, prev = int (e[0]), e[1]
			R[src][n, i+1] += HPUtils.kernel (curr, prev, omega)

		# update the last seen event for the node
		lastseen[src] = (curr, i)

	return R

def multivariateHLLSlow (params, omega, seq, dim, horizon, sign=1.0, verbose=False):
	""" a naive implementation of the likelihood objective function """
	epsilon = 1e-50
	mu = params[0:dim]
	alpha = params[dim:].reshape (dim, dim)
	ll = 0.0
	term1, term2, term3 = 0.0, 0.0, 0.0
	for i in xrange (dim):
		for event in seq:
			intensity = 0.0
			src, curr = int(event[0]), event[1]
			if src == i:
				intensity += mu[i]
				for j in xrange (dim):
					for e in seq:
						s,c = int (e[0]), e[1]
						if c >= curr: break
						if s == j: intensity += alpha[j,i] * HPUtils.kernel (curr, c, omega)
				term1 += np.log (intensity + epsilon)

		for event in seq:
			src, curr = int (event[0]), event[1]
			term3 += (alpha[src,i] * (1 - HPUtils.kernel (horizon, curr, omega)))
	
	term2 += (mu[i] * horizon)
    
	ll = (sign) * (term1 - term2 - term3)
	if verbose: print ll, term1, term2, term3
	return ll

def multivariateHLLGradSlow (params, omega, seq, dim, horizon, R, sign=1.0, verbose=False):
	""" a naive implementation of the gradient """
	epsilon = 1e-50
	mu = params[0:dim]
	alpha = params[dim:].reshape (dim, dim)
	gradmu = np.zeros_like (mu)
	gradalpha = np.zeros_like (alpha)

	for m in xrange (dim):
		gradmu[m] -= horizon
		for i, event in enumerate(seq):
			src, curr = int (event[0]), event[1]
			if src == m:
				intensity = mu[src] + sum([alpha[j,src] * R[src][j,i+1] for j in xrange (dim)]) + epsilon
				gradmu[src] += (1./intensity)
				for j in xrange (dim):
					gradalpha[j,m] += (R[m][j,i+1] / intensity)

	for m in xrange (dim):
		for n in xrange (dim):
			for i, event in enumerate (seq):
				src, curr = int (event[0]), event[1]
				if src == n:
					gradalpha[n,m] -= (1 - HPUtils.kernel (horizon, curr, omega))

	return (sign) * np.concatenate ((gradmu, gradalpha.flatten()))

def multivariateHLL (params, omega, seq, dim, horizon, R, sign=1.0, verbose=False):
	""" computes the multivariate likelihood objective

	Parameters:
	-----------
	R: dict
		key is the node index and value is sparse.lil_matrix of dimension nodes * (events + 1)
	"""
	epsilon = 1e-50
	mu = params[0:dim]
	alpha = params[dim:].reshape (dim, dim)
	term1 = term3 = 0.0
	term2 = np.sum (mu) * horizon
	for i, event in enumerate (seq):
		src, curr = int (event[0]), event[1]
		term1 += np.log (mu[src] + sum([alpha[j,src] * R[src][j,i+1] for j in xrange (dim)]) + epsilon)
		term3 += (alpha[src,:] * (1 - HPUtils.kernel (horizon, curr, omega))).sum()
	ll = (sign) * (term1 - term2 - term3)
	if verbose: print ll, term1, term2, term3
	return ll

def multivariateHLLGrad (params, omega, seq, dim, horizon, R, sign=1.0, verbose=False):
	""" computes the gradient of the multivariate likelihood objective

	Parameters:
	-----------
	R: dict
		key is the node index and value is sparse.lil_matrix of dimension nodes * (events + 1)
	"""
	epsilon = 1e-50
	mu = params[0:dim]
	alpha = params[dim:].reshape (dim, dim)
	gradmu = np.zeros_like (mu)
	gradalpha = np.zeros_like (alpha)

	for i, event in enumerate(seq):
		src, curr = int (event[0]), event[1]
		intensity = mu[src] + sum([alpha[j,src] * R[src][j,i+1] for j in xrange (dim)]) + epsilon
		gradmu[src] += (1./intensity)
		gradalpha[:,src] += (R[src][:,i+1] / intensity).toarray().flatten()
		gradalpha[src, :] -= (1 - HPUtils.kernel (horizon, curr, omega))
	gradmu -= horizon
    
	return (sign) * np.concatenate((gradmu, gradalpha.flatten()))

def multivariate (seq, omega, dim, horizon, verbose=False):
	bounds = [(0,None) for i in xrange (dim + (dim**2))]
	params = np.random.uniform (0, 1, size=dim + dim ** 2)
	R = calculateR (seq, dim, omega)
	res = minimize(multivariateHLL, params, args=(omega, seq, dim, horizon, R, -1.0, verbose), 
				   method="L-BFGS-B", 
				   jac=multivariateHLLGrad, 
				   bounds=bounds,
				   options={"ftol": 1e-10, "maxls": 50, "maxcor":50, "maxiter":100, "maxfun": 100})

	mu = res.x[:dim]
	alpha = res.x[dim:].reshape (dim,dim)
	return mu, alpha

def parametricHLLOld (params, omega, seq, dim, horizon, R, adj, sign=1.0, verbose=False):
	mu = params[0] * np.ones(dim)
	a,b = params[1], params[2]
	alpha = (sp.lil_matrix (np.diag([a] * dim) + b * adj)).tocsr()
	pars = np.concatenate ((mu, alpha.toarray().flatten()))
	return multivariateHLL (pars, omega, seq, dim, horizon, R, sign=sign, verbose=verbose)

def parametricHLLGradOld (params, omega, seq, dim, horizon, R, adj, sign=1.0, verbose=False):
	""" isn't the right thing to do. does not calculate the social parameter's gradient correctly. """
	mu = params[0] * np.ones(dim)
	a,b = params[1], params[2]
	alpha = (sp.lil_matrix (np.diag([a] * dim) + b * adj)).tocsr()

	pars = np.concatenate ((mu, alpha.toarray().flatten()))
	grad = multivariateHLLGrad (pars, omega, seq, dim, horizon, R, sign=sign, verbose=verbose)

	alphagrad = grad[dim:].reshape(dim, dim)
	gradParams = np.zeros_like (params)
	gradParams[0] = grad[0:dim].sum()
	gradParams[1] = np.trace (alphagrad)
	gradParams[2] = np.sum(alphagrad) - gradParams[1]

	return gradParams

def parametricHLL (params, omega, seq, dim, horizon, R, adj, sign=1.0, verbose=False):
	""" computes the parametric multivariate likelihood objective

	Parameters:
	-----------
	R: dict
		key is the node index and value is sparse.lil_matrix of dimension nodes * (events + 1)
	"""
	mu, a, b = params[0], params[1], params[2]
	epsilon = 1e-50
	term1 = term3 = term4 = 0.0
	term2 = mu * dim * horizon
	for i, event in enumerate (seq):
		src, curr = int (event[0]), event[1]
		term1 += np.log (mu + a * R[src][src, i+1] + b * (R[src][:,i+1].multiply(adj[:,src]).sum()) + epsilon)
		decay = (1 - HPUtils.kernel (horizon, curr, omega))
		term3 += decay
		term4 += adj[src,:] * decay
	term4 = term4.sum()
	ll = (sign) * (term1 - term2 - a*term3 - b*term4)
	if verbose: print ll, term1, term2, term3
	return ll
	
def parametricHLLGrad (params, omega, seq, dim, horizon, R, adj, sign=1.0, verbose=False):
	""" computes the parametric multivariate gradient equations

	Parameters:
	-----------
	R: dict
		key is the node index and value is sparse.lil_matrix of dimension nodes * (events + 1)
	""" 
	mu, a, b = params[0], params[1], params[2]
	epsilon = 1e-50

	gradmu = grada = gradb = 0.0
	for i, event in enumerate(seq):
		src, curr = int (event[0]), event[1]
		totalExcitationReceived = R[src][:,i+1].multiply(adj[:,src]).sum()
		intensity = mu + a * R[src][src,i+1] + b * (totalExcitationReceived) + epsilon
		gradmu += (1./intensity)
		decay = 1 - HPUtils.kernel (horizon, curr, omega)
		grada += ((R[src][src,i+1] / intensity ) - decay)
		gradb += (totalExcitationReceived / intensity - (adj[src,:].sum() * decay))
	
	gradmu -= (dim * horizon)
	if verbose: print gradmu, grada, gradb
	return (sign) * np.array ([gradmu, grada, gradb])
	
def parametric (seq, omega, dim, adj, horizon, verbose=False):
	bounds = [(0,None) for i in xrange (3)]
	R = calculateR (seq, dim, omega)
	params = np.random.uniform(0,1,size=3)
	res = minimize(parametricHLL, params, args=(omega, seq, dim, horizon, R, adj, -1.0, verbose), 
				   method="L-BFGS-B",
				   jac=parametricHLLGrad,
				   bounds=bounds,
				   options={"ftol": 1e-10, "maxls": 50, "maxcor":50, "maxiter":1000, "maxfun": 1000})

	return res.x[0], res.x[1], res.x[2]

def customEstimator (init, seq, omega, dim, adj, horizon, objectiveFunction, gradientFunction, verbose=False):
	bounds = [(0, 100) for i in xrange (len(init))]
	if verbose: print "Calculating R ...",
	R = calculateR (seq, dim, omega)
	if verbose: print "done"
	res = minimize (objectiveFunction, init, args=(omega, seq, dim, horizon, R, adj, -1.0, verbose),
					method="L-BFGS-B",
					jac=gradientFunction,
					bounds=bounds,
					options={"ftol": 1e-8, "maxls": 20, "maxcor": 20, "maxiter": 150, "maxfun": 800})
	return res.x

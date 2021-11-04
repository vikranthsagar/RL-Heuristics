import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
# print(sys.executable, sys.path)
import GPy
import pickle
import random
import csv
from scipy.stats import norm,uniform
import copy

class problem(object):

	def __init__(self, ell, v):
		self.ell = ell
		self.v = v
		self.x_min = -10
		self.x_max = 10
		self.DIV = 201
		self.XX = np.linspace(self.x_min,self.x_max, self.DIV)[:, None]
		self.f = copy.deepcopy(self.get_random_function_of_complexity(ell,v))


	def get_random_function_of_complexity(self, ell, v, nugget=1e-6):
		k = GPy.kern.RBF(1, lengthscale=ell, variance = v)
		n = int(20 / ell) * 5
		X = np.linspace(-10, 10, n)[:, None]
		K = k.K(X) + nugget * np.eye(n)
		L = np.linalg.cholesky(K)
		Y = np.dot(L, np.random.randn(n, 1))
		m = GPy.models.GPRegression(X, Y, k)
		m.likelihood.variance = nugget
		f = lambda XX: m.predict(XX)[0]
		return f

	def get_random_function_of_complexity_new(self, ell,v,nugget=1e-6):
		k = GPy.kern.RBF(1, lengthscale=ell, variance=v)
		n = int(20 / ell) * 5
		X = np.linspace(-10, 10, n)[:, None]
		K = k.K(X) + nugget * np.eye(n)
		L = np.linalg.cholesky(K)
		Y = 50+np.dot(L, np.random.randn(n, 1))
		m = GPy.models.GPRegression(X, Y, k)
		m.likelihood.variance = nugget
		f = lambda XX: m.predict(XX)[0]
		Xnew = np.linspace(-10, 10, 201)[:, None]
		F=np.array(0.5*f(Xnew))
		y = F+50*norm.pdf(Xnew, loc=Xnew[np.argmax(F)],scale=0.5)
		y = 100*(y-min(y))/(max(y)-min(y)) #scale from 0 to 100
		y = y.reshape(len(Xnew))
		x = np.linspace(self.x_min, self.x_max, self.DIV)
		g = lambda XX: np.interp(XX, x,y)
		return g

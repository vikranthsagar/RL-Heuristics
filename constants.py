import numpy as np
import scipy
class constants(object):

	def __init__(self, B, c_l, c_h, s_l, s_h):
		self.B = B #Total available budget (20/40/60)
		self.c_l = c_l #Cost of low fidelity simulation (2)
		self.c_h = c_h #Cost of high fidelity simulation (8)
		self.s_l=s_l #Random noise in low fidelity simulation (10)
		self.s_h=s_h #Random noise in high fidelity simulation (0.1)

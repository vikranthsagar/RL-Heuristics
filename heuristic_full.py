import numpy as np
import GPy
import scipy
import random
from scipy import linalg
from sklearn.linear_model import ARDRegression
from scipy.stats import norm, uniform, entropy, logistic, beta, poisson, kstest, probplot, ttest_ind
np.seterr(divide='ignore', invalid='ignore')

class heuristic(object):

    def __init__(self, basis_x, basis_m, basis_s, para_x, para_m, para_s):
        H = list()

        #Adding choose next x sub-heuristic
        switcher_x={
            'maxEI':self.maxEI, #(EU-based) para_x = None
            'UCB': self.UCB, #(Simple heuristic) para_x = None
            'Rand':self.RANDx
        }
        H.append(switcher_x.get(basis_x))

        #Adding choose next m sub-heuristic
        switcher_m={
            'basis_twosteps_ahead':self.basis_twosteps_ahead, #(EU-based) para_m = 0.5 (less than 2)
            'FRB':self.FRB #(Simple heuristic) para_m = 16
            #'basis_sample_number': self.basis_sample_number, #(Simple heuristic)
            #'basis_conditional_mean': self.basis_conditional_mean #(Simple heuristic)
        }
        H.append(switcher_m.get(basis_m))

        #Adding stopping sub-heuristic
        switcher_s={
            'fixed_remaining_budget':self.fixed_remaining_budget, #(Simple heuristic) fraction of some budget
            #'highest_prototype':self.highest_prototype, #(Simple heuristic) #take out
            'rel_maxEI':self.rel_maxEI #(EU-based) para_s = 3 to 5
        }
        H.append(switcher_s.get(basis_s))

        self.H = H
        self.para_x = para_x #Parameter for choose-next-x strategies
        self.para_m = para_m #Parameter for choose-next-model strategies
        self.para_s = para_s #Parameter for stopping stratregies


    #Choose next x heuristics
    def maxEI(self, x, y, m, i, B, mu_i, sig_i, my_cues, diff=None):
        # mu_max=max(mu_i[[int(10*(x_t+10)) for x_t in x]])
        if len(mu_i) != 1:
            EI = my_cues.get_EI(mu_i, sig_i, max(mu_i))
        else:
            EI = np.array([1])
        XX = my_cues.XX
        return XX[np.argmax(EI)][0]

    def UCB(self, x, y, m, i, B, mu_i, sig_i, my_cues, diff=None):
        i = np.argmax(mu_i + 1.96 * sig_i)
        ucb = max(mu_i + 1.96*sig_i)
        # print((mu_i + 1.96*sig_i).T,ucb)
        # print(ucb)
        XX = my_cues.XX
        return XX[i][0]

    def RANDx(self, x, y, m, i, B, mu_i, sig_i, my_cues, diff=None):
        XX = my_cues.XX
        rand_x = random.choice(XX)
        while rand_x in x:
            rand_x = random.choice(XX)

        return rand_x[0]

    #Choosing next information source
    def basis_twosteps_ahead(self, x, y, m, i, B, mu_i, sig_i, my_cues, diff=None):
        #Return whether or not to sample high fidelity simulation
        para_m = self.para_m
        #
        if para_m == 0:
            # print('m = %1.1f'%para_m)
            return np.array([1])<para_m
        elif diff:
            return diff<para_m
        else:
            mu_max = max(mu_i[[int(10*(x_t+10)) for x_t in x]])
            EI = my_cues.get_EI(mu_i, sig_i, max(mu_i))
            XX = my_cues.XX
            m = np.append(m, 0)
            ECI=[]
            for i in range(XX.shape[0]):
                x_temp = np.vstack((x, [XX[i]]))
                y_temp = np.vstack((y, [[0]]))
                trash, sig_temp = my_cues.set_multinoise_uncertainty(x_temp, m, y_temp)
                tem=my_cues.get_EI(mu_i, sig_temp, max(mu_i))
                ECI.append(max(tem))

            diff = max(EI+ECI)-max(EI)
            return diff<para_m

    def basis_sample_number(self, x, y, m, i, B, mu_i, sig_i, my_cues, diff=None):
        para_m = self.para_m
        return i>para_m

    def basis_conditional_mean(self, x, y, m, i, B, mu_i, sig_i, my_cues, diff=None):
        if diff:
            return diff
        max_mu = max(mu_i)
        mu_x = mu_i[int(10*(x[i-1]+10))]
        para_m = self.para_m
        return (max_mu-mu_x) < para_m

    def FRB(self, x, y, m, i, B, mu_i, sig_i, my_cues, diff=None):
        para_m = self.para_m
        if para_m ==0:
            # print('m = %1.1f'%para_m)
            return np.array([1])<para_m
        else:
            cost = np.zeros(m.shape)
            cost[m==0]=my_cues.c_l;cost[m==1]=my_cues.c_h;
            diff = B - sum(cost)
            # print ('RB : %2.3f'%diff) ###
            return diff < para_m
    #Stopping heuristics, return whether to stop or not
    def fixed_remaining_budget(self, x, y, m, i, B, mu_i, sig_i, my_cues,  Max_EI, diff=None):
        cost = np.zeros(m.shape)
        cost[m==0]=my_cues.c_l;cost[m==1]=my_cues.c_h;
        diff = B - sum(cost)
        para_s = self.para_s
        #print ('RB : %2.3f'%diff) ###
        return (diff < para_s)[0], np.array((1,1))

    def highest_prototype(self, x, y, m, i, B, mu_i, sig_i, my_cues,  Max_EI, diff=None):
        if np.any(m==1):
            max_yh=max(y[m==1])
        else:
            max_yh=0
        diff = max_yh-max(mu_i)
        para_s = self.para_s
        return diff < para_s

    def rel_maxEI(self, x, y, m, i, B, mu_i, sig_i, my_cues, Max_EI, diff=None):
        # mu_max=max(mu_i[[int(10*(x_t+10)) for x_t in x]])
        if len(mu_i) != 1:
            EI = my_cues.get_EI(mu_i, sig_i, max(mu_i))
            para_s = self.para_s
            # if len(Max_EI)==1:
            # 	max_ei = 1
            # else:
            # 	max_ei = Max_EI[1]
            # print(max(EI)/max_ei)
            # print('Max_EI: %2.3f'%max(EI)) ###
            # print('rel_ei: %2.3f'%rel_ei) ###

            if i <= 2:
                s = False
            else:
                rei = max(EI)[0]/Max_EI[0]
                s = rei<para_s
            # s = rei<para_s
            return s , max(EI)
        else:
            return False, np.array([1])


class cues(object):

    def __init__(self,  my_const, obj_prob):
        self.c_l=my_const.c_l
        self.c_h=my_const.c_h
        self.s_l=my_const.s_l
        self.s_h=my_const.s_h
        self.x_min = obj_prob.x_min
        self.x_max = obj_prob.x_max
        self.DIV = obj_prob.DIV
        self.XX =obj_prob.XX
        self.ell = obj_prob.ell
        self.v = obj_prob.v

    def set_multinoise_uncertainty(self, X, fidelity, Y, mean=0., return_COV=False, model=0):
    #     X = np.array([float(e) for e in x_string.split(",")])[:, None]66
    #     Y = np.array([float(e) for e in y_string.split(",")])[:, None]
    #     fidelity = np.array(fidelity_string.split(","))
        # noise=[]
        # for item in fidelity:
        # 	if item==0:
        # 		noise.append(self.s_l)
        # 	else:
        # 		noise.append(self.s_h)

        # Noise = np.diag(np.square(noise))
        if model == 0:
            k = GPy.kern.RBF(input_dim=1, lengthscale=self.ell, variance=self.v)
        # k_fixed = GPy.kern.Fixed(1, GPy.util.linalg.tdot(np.array(noise)[:, None]))
            m = GPy.models.GPRegression(X, Y, k)
            m.likelihood.variance.constrain_fixed(1e-16) #assuming no measurment error
            Y_p, V_p = m.predict(self.XX)
            S_p = np.sqrt(V_p)
        # Lower predictive bound
            Y_l = Y_p - 2. * S_p
        # Upper predictive bound
            Y_u = Y_p + 2. * S_p

        if model ==1: # lienar
            phi = LinearBasis()
            Phi = compute_design_matrix(X, phi)
            regressor = ARDRegression()
            regressor.fit(Phi, np.ravel(Y))
            sigma = np.sqrt(1. / regressor.alpha_)
            alpha = regressor.lambda_
            A = np.dot(Phi.T, Phi) / sigma ** 2. + alpha * np.eye(Phi.shape[1])
            L = scipy.linalg.cho_factor(A)
            m = scipy.linalg.cho_solve(L, np.dot(Phi.T, Y) / sigma ** 2)  # The posterior mean of w
            S = scipy.linalg.cho_solve(L, np.eye(Phi.shape[1]))           # The posterior covariance of w
            Phi_p = compute_design_matrix(self.XX, phi)
            Y_p = np.dot(Phi_p, m) # The mean prediction
            V_p_ep = np.einsum('ij,jk,ik->i', Phi_p, S, Phi_p)
            V_p = V_p_ep + sigma ** 2 # Full uncertainty
            S_p = np.sqrt(V_p)[:,None]

        if model ==2:
            phi = PolynomialBasis(2)
            Phi = compute_design_matrix(X, phi)
            regressor = ARDRegression()
            regressor.fit(Phi, np.ravel(Y))
            sigma = np.sqrt(1. / regressor.alpha_)
            alpha = regressor.lambda_
            A = np.dot(Phi.T, Phi) / sigma ** 2. + alpha * np.eye(Phi.shape[1])
            L = scipy.linalg.cho_factor(A)
            m = scipy.linalg.cho_solve(L, np.dot(Phi.T, Y) / sigma ** 2)  # The posterior mean of w
            S = scipy.linalg.cho_solve(L, np.eye(Phi.shape[1]))           # The posterior covariance of w
            Phi_p = compute_design_matrix(self.XX, phi)
            Y_p = np.dot(Phi_p, m) # The mean prediction
            V_p_ep = np.einsum('ij,jk,ik->i', Phi_p, S, Phi_p)
            V_p = V_p_ep + sigma ** 2 # Full uncertainty
            S_p = np.sqrt(V_p)[:,None]

        if model ==3:
            phi = PolynomialBasis(3)
            Phi = compute_design_matrix(X, phi)
            regressor = ARDRegression()
            regressor.fit(Phi, np.ravel(Y))
            sigma = np.sqrt(1. / regressor.alpha_)
            alpha = regressor.lambda_
            A = np.dot(Phi.T, Phi) / sigma ** 2. + alpha * np.eye(Phi.shape[1])
            L = scipy.linalg.cho_factor(A)
            m = scipy.linalg.cho_solve(L, np.dot(Phi.T, Y) / sigma ** 2)  # The posterior mean of w
            S = scipy.linalg.cho_solve(L, np.eye(Phi.shape[1]))           # The posterior covariance of w
            Phi_p = compute_design_matrix(self.XX, phi)
            Y_p = np.dot(Phi_p, m) # The mean prediction
            V_p_ep = np.einsum('ij,jk,ik->i', Phi_p, S, Phi_p)
            V_p = V_p_ep + sigma ** 2 # Full uncertainty
            S_p = np.sqrt(V_p)[:,None]

        return Y_p, S_p
        # Construct the GP regression model
    #     s = 2.
        # K = k.K(X)
        # cho = scipy.linalg.cho_factor(K + Noise)

        # Now we are ready to make predictions
        # Xp = np.linspace(Constants.min_x, Constants.max_x, Constants.DIVISIONS)[:, None]
        # Kp = k.K(self.XX)
        # cK = k.K(self.XX, X)
        # Predictive mean
        # if mean:
        # 	alpha = scipy.linalg.cho_solve(cho, (Y-mean))
        # 	m_tilde = mean + np.dot(cK, alpha)
        # else:
        # 	alpha = scipy.linalg.cho_solve(cho, Y)
        # 	m_tilde = np.dot(cK, alpha)
        #
        # tmp = scipy.linalg.cho_solve(cho, cK.T)
        # # Predictive variance
        # s_tilde = np.sqrt(np.diag(Kp) - np.diag(np.dot(cK, tmp)))[:, None]
        #
        # if return_COV:
        # 	return m_tilde, s_tilde, Kp - np.dot(cK, tmp) + 1e-06 * np.eye(self.XX.shape[0])
        # else:
            # return m_tilde, s_tilde

    def set_multinoise_uncertainty2(ndim, ell, X, fidelity, Y, mean=0., return_COV=False):
        noise=[]
        for item in fidelity:
            if item==0:
                noise.append(1)
            else:
                noise.append(0.1)
        Noise = np.diag(np.square(noise))
        k = GPy.kern.RBF(input_dim=ndim, lengthscale=ell, variance=1)
        k_fixed = GPy.kern.Fixed(2, GPy.util.linalg.tdot(np.array(noise)[:, None]))
        m = GPy.models.GPRegression(X, Y-mean, k+k_fixed)
        XX = np.linspace(-10, 10, 201)[:, None]
        xv, yv = np.meshgrid(XX,XX)
        XX = np.vstack([xv.flatten(), yv.flatten()]).T
        Y_p, V_p = m.predict(XX, kern=k.copy())
        S_p = np.sqrt(V_p)
        return Y_p+mean, S_p

    def get_EI(self, Y_p, V_p, f_max):
#         f_max=max(Y_p)
        u = (Y_p - f_max) / V_p
        ei = V_p * (u * norm.cdf(u) + norm.pdf(u))
        ei[V_p <= 0.] = 0.
        # EI = (Y_p-f_max)*norm.cdf((Y_p-f_max)/V_p) + V_p*norm.pdf((Y_p-f_max)/V_p)
        return ei

    def get_PI(self, Y_p, V_p, f_max):
        PI=[]
        for i in range(Y_p.shape[0]):
            PI.append(1-norm.cdf(f_max, loc=Y_p[i], scale=np.sqrt(V_p[i])))
        return np.array(PI)

    def get_ECI(self, X_obs, fidelity, Y_obs, Y_p, V_p):
        X_p = self.XX
        X_obs = X_obs[:, None]
        Y_obs = Y_obs[:, None]
        fidelity = fidelity[:, None]

        ECImax = []
        ECImin = []
        fidelity=np.append(fidelity, 0)
        for i in range(X_p.shape[0]):
            X_temp = np.vstack((X_obs, [X_p[i]]))
            Y_temp = np.vstack((Y_obs, [[0]]))
            trash, V_p_temp = self.set_multinoise_uncertainty(X_temp, fidelity, Y_temp)
            tem=self.get_EI(Y_p, V_p_temp, max(Y_p))
            ECImax.append(max(tem))
            ECImin.append(-1*np.trapz(tem.flatten(), X_p.flatten()))
        return ECImax,ECImin
##################################
class LinearBasis(object):
    """
    Represents a 1D linear basis.
    """
    def __init__(self):
        self.num_basis = 2 # The number of basis functions
    def __call__(self, x):
        """
        ``x`` should be a 1D array.
        """
        return [1., x[0]]
#################################
class PolynomialBasis(object):
    """
    A set of linear basis functions.

    Arguments:
    degree  -  The degree of the polynomial.
    """
    def __init__(self, degree):
        self.degree = degree
        self.num_basis = degree + 1
    def __call__(self, x):
        return np.array([x[0] ** i for i in range(self.degree + 1)])
##################################
class RadialBasisFunctions(object):
    """
    A set of linear basis functions.

    Arguments:
    X   -  The centers of the radial basis functions.
    ell -  The assumed lengthscale.
    """
    def __init__(self, X, ell):
        self.X = X
        self.ell = ell
        self.num_basis = X.shape[0]
    def __call__(self, x):
        return np.exp(-.5 * (x - self.X) ** 2 / self.ell ** 2).flatten()

###############################################
def compute_design_matrix(X, phi):
    """
    Arguments:

    X   -  The observed inputs (1D array)
    phi -  The basis functions.
    """
    num_observations = X.shape[0]
    num_basis = phi.num_basis
    Phi = np.ndarray((num_observations, num_basis))
    for i in range(num_observations):
        Phi[i, :] = phi(X[i, :])
    return Phi

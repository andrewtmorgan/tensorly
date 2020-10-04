import numpy as np
from ..base import partial_tensor_to_vec, partial_unfold
from ..tenalg import  khatri_rao
from ..kruskal_tensor import kruskal_to_tensor, kruskal_to_vec
from ..random import check_random_state
from .. import backend as T
from lnpy import linear as ln
importe lmfit
from tqdm import tqdm
from tqdm import trange


# Author: Jean Kossaifi

# License: BSD 3 clause



class KruskalRegressor():
    """Kruskal tensor regression

        Learns a low rank CP tensor weight

    Parameters
    ----------
    weight_rank : int
        rank of the CP decomposition of the regression weights
    tol : float
        convergence value
    reg_W : int, optional, default is 1
        regularisation on the weights
    n_iter_max : int, optional, default is 100
        maximum number of iteration
    random_state : None, int or RandomState, optional, default is None
    verbose : int, default is 1
        level of verbosity
    """

    def __init__(self, weight_rank, tol=10e-7, reg_W=1, n_iter_max=100, random_state=None, verbose=1):
        self.weight_rank = weight_rank
        self.tol = tol
        self.reg_W = reg_W
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, **kwargs):
        """Returns a dictionary of parameters
        """
        params = ['weight_rank', 'tol', 'reg_W', 'n_iter_max', 'random_state', 'verbose']
        return {param_name: getattr(self, param_name) for param_name in params}

    def set_params(self, **parameters):
        """Sets the value of the provided parameters"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, W=None, D=np.zeros((8,1))):
        """Fits the model to the data (X, y)

        Parameters
        ----------
        X : ndarray
            tensor data of shape (n_samples, N1, ..., NS)
        y : 1D-array of shape (n_samples, )
            labels associated with each sample
        W : list
            list of initial covariates (S-entry list; each entry the size of N1, ..., NS)

        Returns
        -------
        self
        """
        if W==None:  # Initialize randomly if W is not provided
            rng = check_random_state(self.random_state)

            # Initialise randomly the weights
            W = []
            for i in range(1, T.ndim(X)):  # The first dimension of X is the number of samples
                W.append(T.tensor(rng.randn(X.shape[i], self.weight_rank), **T.context(X)))

        # Norm of the weight tensor at each iteration
        norm_W = []
        weights = T.ones(self.weight_rank, **T.context(X))

        # Need an intercept
        intercept = 0.0

        # functions for fitting spatial gaussians
        def solve_ald(x, y, cen, sig, rho, delta, nv):

            tmp_gauss = gaussian(np.arange(0, 27, 0.5), cen, sig, 1)
            cov_sp = np.zeros((27,27))
            for cc in range(27):
                np.fill_diagonal(cov_sp[cc:], tmp_gauss[cc:-cc:2])
                np.fill_diagonal(cov_sp[:,cc:], tmp_gauss[cc:-cc:2])
            np.fill_diagonal(cov_sp, tmp_gauss[::2])

            dist_mat = np.zeros((27,27))
            for cc in range(1, 27):
                np.fill_diagonal(dist_mat[cc:], np.ones((27-cc, 1))*(cc**2))
                np.fill_diagonal(dist_mat[:,cc:], np.ones((27-cc, 1))*(cc**2))
            cov_sm = np.exp(-rho - (dist_mat / delta**2))

            cov = cov_sm * cov_sp
            npad = ((0, 1), (0, 1))
            cov = np.pad(cov, pad_width=npad, mode='constant', constant_values=0)

            x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            CXX = cov.dot(x.T).dot(x)
            XY = x.T.dot(y)
            YY = y.T.dot(y)

            lamb = np.linalg.solve((1./nv)*CXX + np.eye(28), cov)
            lamb_inv = np.linalg.pinv(lamb)
            mu = np.linalg.solve(CXX + nv*np.eye(28) , cov).dot(XY)

            return mu, lamb, lamb_inv, cov


        def logevid(pars, x, y):

            # unpack parameters: extract .value attribute for each parameter
            parvals = pars.valuesdict()
            cen = parvals['cen']
            sig = parvals['sig']
            rho = parvals['rho']
            delta = parvals['delta']
            nv = parvals['nv']

            mu, lamb, lamb_inv, cov = solve_ald(x, y, cen, sig, rho, delta, nv)
            YY = y.T.dot(y)

            evid = (-y.size/2)*np.log(2*np.pi*nv) - \
                   0.5 * np.log(cov.dot(lamb_inv)) + \
                   0.5 * mu.T.dot(lamb_inv).dot(mu) - \
                   (1/(2*nv)) * YY
            return -evid
        

        def gaussian(x, mu, sig, sc):
            tmp = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
            return tmp*sc


        for iteration in trange(self.n_iter_max):

            # Optimise each factor of W
            for i in range(len(W)):
                phi = T.reshape(
                          T.dot(partial_unfold(X, i, skip_begin=1),
                                khatri_rao(W, skip_matrix=i)),
                      (X.shape[0], -1))
                
                #  
                if i==0:  # There should be a better way
                    ridge = ln.SmoothRidge(D=D,
                                           verbose=False)
                    ridge.fit(phi, y)
                    W[i] = ridge.coef_[:, None]
                    intercept = ridge.intercept_ 

                elif i==1:
                    ridge = ln.SmoothRidge(D=(phi.shape[1], 1),
                                           second_order=False,
                                           verbose=False)
                    ridge.fit(phi, y)
                    W[i] = ridge.coef_[:, None]
                    intercept = ridge.intercept_ 

                elif i<4:
                    # get initial
                    ridge = ln.Ridge(verbose=False)
                    ridge.fit(phi, y)

                    # set parameters
                    params = lmfit.Parameters()
                    params['cen'] = lmfit.Parameter(value=13, min=7, max=19)
                    params['sig'] = lmfit.Parameter(value=4, min=2, max=9)
                    params['rho'] = lmfit.Parameter(value=-np.log(ridge.alpha), min=-20, max=20)
                    params['delta'] = lmfit.Parameter(value=1., min=-10e6, max=10e6)
                    params['nv'] = lmfit.Parameter(value=ridge.noisevar, min=10e-7, max=10)
                    
                    # solve
                    out = lmfit.minimize(logevid, params, args=(phi, y))
                    mu, _, _, _ = solve_ald(phi, y, out.params['cen'].value,
                                                    out.params['sig'].value,
                                                    out.params['rho'].value, 
                                                    out.params['delta'].value,
                                                    out.params['nv'].value)
                    W[i] = mu[:-1]

                else:
                    ridge = ln.SmoothRidge(D=(phi.shape[1], 1),
                                           zero_order=True,
                                           first_order=True,
                                           second_order=True,
                                           verbose=False)
                    ridge.fit(phi, y)
                    W[i] = ridge.coef_[:, None]
                    intercept = ridge.intercept_ 
                

            weight_tensor_ = kruskal_to_tensor((weights, W))
            norm_W.append(T.norm(weight_tensor_, 2))

            # Convergence check
            if iteration > 1:
                weight_evolution = abs(norm_W[-1] - norm_W[-2]) / norm_W[-1]

                if (weight_evolution <= self.tol):
                    if self.verbose:
                        print('\nConverged in {} iterations'.format(iteration))
                    break

        self.weight_tensor_ = weight_tensor_
        self.kruskal_weight_ = (weights, W)
        self.intercept_ = intercept

        self.vec_W_ = kruskal_to_vec((weights, W))
        self.n_iterations_ = iteration + 1
        self.norm_W_ = norm_W

        return self

    def predict(self, X):
        """Returns the predicted labels for a new data tensor

        Parameters
        ----------
        X : ndarray
            tensor data of shape (n_samples, N1, ..., NS)
        """
        return T.dot(partial_tensor_to_vec(X), self.vec_W_)

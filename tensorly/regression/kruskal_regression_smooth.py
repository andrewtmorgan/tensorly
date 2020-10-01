import numpy as np
from ..base import partial_tensor_to_vec, partial_unfold
from ..tenalg import  khatri_rao
from ..kruskal_tensor import kruskal_to_tensor, kruskal_to_vec
from ..random import check_random_state
from .. import backend as T
from lnpy import linear as ln

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

        for iteration in range(self.n_iter_max):

            # Optimise each factor of W
            for i in range(len(W)):
                phi = T.reshape(
                          T.dot(partial_unfold(X, i, skip_begin=1),
                                khatri_rao(W, skip_matrix=i)),
                      (X.shape[0], -1))
                
                # 
                if phi.shape[1]==8:  # There should be a better way
                    asd = ln.ASD(D=D,
                                 init_coef=W[i].squeeze(),
                                 init_intercept=0,
                                 verbose=False)
                else:
                    asd = ln.ASD(D=(phi.shape[1], 1),
                                 init_coef=W[i].squeeze(),
                                 init_intercept=0,
                                 verbose=False)
                asd.fit(phi, y)
                W[i] = asd.coef_[:, None]

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
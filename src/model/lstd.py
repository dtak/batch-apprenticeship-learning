# inspired by: https://github.com/stober/lspi/blob/master/src/lspi.py
# heavily modified
import time
import itertools
import logging
import sys

import numpy as np
from numpy.linalg import inv, norm, cond, solve, matrix_rank, lstsq

import scipy.sparse.linalg as spla
from numba import jit, jitclass, int32, float32
import multiprocessing

from core.policy import LinearQ2
from contrib.baselines import logger

@jit
def fast_solve(phi_sa, phi_sa_next, r, gamma):
    # n x p matrix
    # n x p matrix
    phi_delta = phi_sa - gamma * phi_sa_next
    # rows where absorb is true should be zero
    # A_hat: p x p matrix
    A_hat = phi_sa.T.dot(phi_delta)
    # b_hat: p x 1 matrix
    b_hat = phi_sa.T.dot(r)
    return A_hat, b_hat

def fast_choose_action(action_list, w, phi):
    def f(s):
        q_list = []
        for a in action_list:
            q = np.asscalar(w.T.dot(phi(s, a).T))
            q_list.append(q)
        q_list = np.array(q_list)
        ties = np.flatnonzero(q_list == q_list.max())
        return np.random.choice(ties)
    return f


class LSTDQ(object):
    """Docstring for LSTD. """

    def __init__(self, p, gamma, lstd_eps):
        """TODO: to be defined1.

        Parameters
        ----------
        D : trajectory data
        p : dimension of phi
        gamma : (0, 1)
        eps : small positive value to make A invertible
        reward_fn : (optional) non-default reward fn to simulate MDP\R
        """

        self._p = p
        self._gamma = gamma
        self._lstd_eps = lstd_eps
        self._W_hat = None

    def fit2(self, phi_sa, phi_sa_next, r):
        """TODO: Docstring for learn.

        assuminng action-value function Q(s,a)
        is linearly parametrized by W
        such that Q = W^T phi(s)

        Parameters
        ----------

        Returns
        -------
        TODO
        this is LSTD_Q
        check dimensionality of everything

        """

        A_hat = self._lstd_eps * np.identity(self._p)

        phi_delta = phi_sa - self._gamma * phi_sa_next
        A_hat += phi_sa.T.dot(phi_delta)
        b_hat = phi_sa.T.dot(r)

        rank = matrix_rank(A_hat)
        if rank == self._p:
            W_hat = solve(A_hat, b_hat)
        else:
            logger.log("condition number of A_hat\n{}".format(cond(A_hat)))
            logger.log("A_hat is not full rank {} < {}".format(rank, self._p))
            W_hat = lstsq(A_hat, b_hat)[0]

        self._W_hat = W_hat
        return W_hat

    def fit(self, phi_sa, phi_sa_next, r):
        """TODO: Docstring for learn.

        assuminng action-value function Q(s,a)
        is linearly parametrized by W
        such that Q = W^T phi(s)

        Parameters
        ----------

        Returns
        -------
        TODO
        this is LSTD_Q
        check dimensionality of everything

        """
        gamma = self._gamma
        A_hat, b_hat = fast_solve(phi_sa, phi_sa_next, r, gamma)

        rank = matrix_rank(A_hat)
        if rank == self._p:
            W_hat = solve(A_hat, b_hat)
        else:
            logger.log("condition number of A_hat\n{}".format(cond(A_hat)))
            logger.log("A_hat is not full rank {} < {}".format(rank, self._p))
            W_hat = lstsq(A_hat, b_hat)[0]

        self._W_hat = W_hat
        return W_hat


    def estimate_Q(self, s0, a0):
        """estimate Q^pi(s,a)

        essentially policy evaluation

        Parameters
        ----------

        Returns
        -------
        Q_hat : Q estimate given a fixed policy pi
        """
        return self._W_hat.T.dot(self.phi(s0, a0))


class LSTDMu(object):
    """Docstring for LSTDMu. """

    def __init__(self, p, q, gamma, lstd_eps, stochastic):
        """TODO: to be defined1.

        Parameters
        ----------
        D : trajectory data
        p : dimension of phi
        phi : basis function for reward
        psi : basis function for feature expectation
        gamma : (0, 1)
        eps : small positive value to make A invertible
        W : W to evaluate
        """
        self._p = p
        self._gamma = gamma
        self._lstd_eps = lstd_eps
        self._q = q
        self._xi_hat = None
        self._stochastic = stochastic


    def fit(self, D, pi):
        """estimate xi to compute mu

        assuminng action-value function mu(s, a)
        is linearly parametrized by xi
        such that mu(s, a) = Q_phi(s, a) = xi^T psi(s)

        Parameters
        ----------
        pi : Policy
            policy to evaluate

        Returns
        -------
        xi_hat = xi_hat

        TODO
        - vectorize this
        - phi(s, a) or phi(s) when to use
        - what phi or psi to use?
        - check dimensionality of everytthing


        """
        self._D = D

        s_next = self._D["s_next"]
        absorb = self._D["done"]
        phi_sa = self._D["phi_sa"]

        psi_sa = self._D["psi_sa"]
        self._psi = self._D["psi_fn"]

        a_next = [pi.act(self._stochastic, s[np.newaxis,...])[0] for s in s_next]

        psi_sa_next = self._psi(s_next, a_next)
        psi_sa_next[absorb.flatten(), :] = 0

        A_hat = np.zeros((self._q, self._q))
        A_hat += self._lstd_eps * np.identity(self._q)
        b_hat = np.zeros((self._q, self._p))

        psi_delta = psi_sa - self._gamma * psi_sa_next

        A_hat += psi_sa.T.dot(psi_delta)
        b_hat = psi_sa.T.dot(phi_sa)

        rank = matrix_rank(A_hat)
        if rank == self._p:
            xi_hat = solve(A_hat, b_hat)
        else:
            logger.log("condition number of A_hat\n{}".format(cond(A_hat)))
            logging.warning("A_hat is not full rank {} < {}".format(rank, self._p))
            xi_hat = lstsq(A_hat, b_hat)[0]

        self._xi_hat = xi_hat
        return xi_hat

    def predict(self, s0, a0):
        """estimate mu

        Parameters
        ----------

        Returns
        -------
        mu_hat = mu_hat

        TODO
        - what if no action?
        """

        return self._xi_hat.T.dot(self._psi(s0, a0)[0].T)


class LSPI(object):
    """Docstring for LSPI. """
    def __init__(self,
                 D,
                 action_list,
                 p,
                 gamma,
                 precision,
                 lstd_eps,
                 W_0,
                 reward_fn,
                 stochastic,
                 max_iter=3):
        """TODO: to be defined1.

        Parameters
        ----------
        D : TODO
        action)list : list of valid action indices
        collet_D : fn that collects extra samples
        p : dimension of phi
        phi : TODO
        gamma : TODO
        precision : convergence threshold
        eps : make A invertible
        W_0 : initial weight
        reward_fn : (optional) non-default reward fn to simulate MDP\R
        max_iter : int
            The maximum number of iterations force termination.
        """
        self._D = D
        self._stochastic = stochastic
        self._action_list = action_list
        self._p = p
        self._gamma = gamma
        self._precision = precision
        self._lstd_eps = lstd_eps
        self._W_0 = W_0
        self._W = None
        self._reward_fn = reward_fn
        self._max_iter = max_iter


    def solve(self, reward_fn=None, return_w_trace=False, return_policy=False):
        """
        allow to define reward fn here also
        """
        self._reward_fn = reward_fn

        W = self._W_0

        W_list = []

        s = self._D["s"]
        a = self._D["a"]
        s_next = self._D["s_next"]
        absorb = self._D["done"]
        phi_sa = self._D["phi_sa"]
        phi = self._D["phi_fn"]


        if self._reward_fn is not None:
            r = np.vstack([self._reward_fn(s,a) for s, a in zip(s, a)])
            logging.info("reward:\nmax:{}\nmin:{}\nmean:{}\n".format(r.max(), r.min(), r.mean()))
        else:
            r = self._D["r"]


        for t in itertools.count():

            W_old = W
            lstd_q = LSTDQ(p=self._p,
                           gamma=self._gamma,
                           lstd_eps=self._lstd_eps)

            pi = LinearQ2(action_list=self._action_list,
                          W=W_old,
                          phi=phi)

            a_next = [pi.act(self._stochastic, s)[0] for s in s_next]
            a_next = np.expand_dims(a_next, axis=1)
            phi_sa_next = phi(s_next, a_next)
            phi_sa_next[absorb.flatten(), :] = 0

            W = lstd_q.fit(phi_sa, phi_sa_next, r)

            W_list.append(W)

            #logging.info("lspi norm {}".format(norm(W - W_old, 2)))
            if t > self._max_iter or norm(W - W_old) < self._precision:
                break

        self._W = W

        if return_w_trace:
            return W, W_list
        elif return_policy:
            pi = LinearQ2(action_list=self._action_list,
                          W=W,
                          phi=phi)
            return pi
        else:
            return W






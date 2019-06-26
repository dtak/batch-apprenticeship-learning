import itertools
import numpy as np
import tensorflow as tf

import os
import sys
path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(path, os.pardir))

import tensorflow as tf
import logging

from model.lstd import LSTDMu
from model.dsfn import DeepSucessorFeatureNetwork as DSFN

import abc
from abc import ABC, abstractmethod

class MuEstimator(ABC):
    """Docstring for MuEstimator. """

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def estimate(self):
        pass


class MCMuEstimator(MuEstimator):
    """Docstring for MCMuEstimator. """

    def __init__(self, phi, gamma):
        """TODO: to be defined1. """
        self._phi = phi
        self._gamma = gamma

    def fit(self, env, pi_eval, stochastic):
        self._env = env
        self._pi_eval = pi_eval
        self._stochastic = stochastic

    def estimate(self, s, a, n_sample=10):
        """
        using monte carlo samples with a simulator
        """
        mus = []
        for epi_i in range(n_sample):
            # this is not fixed
            s = self._env.reset()
            mu = 0.0
            for t in itertools.count():
                a = self._pi_eval.act(self._stochsatic, s)[0]
                s_next, r, done, _ = self._env.step(a)
                mu += self._gamma ** t * self._phi(s, a).flatten()
                s = s_next
                if done:
                    break
            mus.append(mu)
        return np.array(mus).mean(axis=0)


class EmpiricalMuEstimator(MuEstimator):
    """Docstring for MCMuEstimator. """

    def __init__(self, phi, gamma):
        """TODO: to be defined1. """
        self._phi = phi
        self._gamma = gamma

    def fit(self, D, stochastic, return_s_init=True):
        """

        Parameters
        ----------
        D : list
            state trajectories


        phi : function
            basis function for features

        gamma : float
            discount factor

        return_s_init : bool
            whether to return initial state list (useful for mu-based IRL algos)
        """
        self._D = D
        self._stochastic = stochastic
        self._return_s_init = return_s_init

    def estimate(self):
        """
        with batch data (mc samples)

        Returns
        -------
        numpy.array

        """
        D = self._D
        mus = []
        s_init = []
        mu = 0.0
        t = 0
        is_s_init = True


        for t, (s, a, done) in enumerate(zip(D["s"], D["a"], D["absorb"])):
            if is_s_init:
                s_init.append(s)
                is_s_init = False
            mu += self._gamma ** t * self._phi(s, a).flatten()
            if done:
                mus.append(mu)
                mu = 0.0
                t = 0
                is_s_init = True

        mu_est = np.array(mus).mean(axis=0)

        if self._return_s_init:
            return mu_est, s_init

        return mu_est


class LSTDMuEstimator(MuEstimator):
    """
    implements LSTD-mu framework (Edouard Klein, 2011)

    this is a deterministic estimator, perhaps it may make
    sense to add small Gaussian noise for robustness.

    Estimates E[mu(s_0)]
    """

    def __init__(self, phi, gamma, D, p, q, eps, s_init_list):
        self._phi = phi
        self._gamma = gamma
        self._D = D
        self._p = p
        self._q = q
        self._eps = eps

        # minor technicality here
        # we need to marginalize the initial state out
        # \sum_s0 E[mu(s0, a) | pi] for all empirical s0
        # could be weighted by empirical distribution of d(s0)
        # but we assume s0 ~ unif(S0)
        # include only unique s_0
        s0_tup_list = [tuple(s) for s in s_init_list]
        self._snl = [np.array(s) for s in set(s0_tup_list)]

    def fit(self, pi_eval, stochastic):
        self._pi_eval = pi_eval
        self._stochastic = stochastic
        self._estimator = LSTDMu(self._p, self._q, self._gamma, self._eps,
                self._stochastic)
        self._estimator.fit(D=self._D, pi=pi_eval)

    def _estimate(self, s, a, n_sample=5):
        mu_list = []

        # isotropic
        # assumes features are standardized
        cov = np.identity(self._p) * 0.05

        mean = self._estimator.predict(s, a)
        mu = np.random.multivariate_normal(mean, cov, size=n_sample)
        mu_list.append(mu.mean(axis=0))

        mu_hat = np.array(mu_list).mean(axis=0)
        return mu_hat


    def estimate(self, s=None, a=None, n_sample=5):
        """TODO: Docstring for estimate.

        Parameters
        ----------
        s : TODO
        a : TODO
        n_sample : TODO, optional

        Returns
        -------
        TODO

        """

        f = self._estimate
        pa = self._pi_eval.act

        if s is not None and a is not None:
            return f(s, a, n_sample)
        if s is not None:
            a = pa(self._stochastic, s)[0]
            return f(s, a, n_sample)
        elif s is None and a is None:

            mu_est_list = []
            for s in self._snl:
                # we don't marginalize action out (we can)
                a = pa(self._stochastic, s[np.newaxis,...])[0]
                mu_est_list.append(f(s, a, n_sample))
            return np.mean(mu_est_list, axis=0)
        else:
            raise Exception("Not Ready")


class DeepMuEstimator(MuEstimator):
    """

       Estimates E[mu(s_0)]

       Parameters
       ----------

       D : dict

       s_init_list : list
           initial states empirically observed

    """
    def __init__(self, phi, gamma, D_train, D_val, s_init_list, ob_space, ac_space, mu_dim, horizon):
        self._phi = phi
        self._gamma = gamma
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._horizon = horizon
        self._mu_dim = mu_dim
        self._D_train = D_train
        self._D_val = D_val

        # minor technicality here
        # we need to marginalize the initial state out
        # \sum_s0 E[mu(s0, a) | pi] for all empirical s0
        # could be weighted by empirical distribution of d(s0)
        # but we assume s0 ~ unif(S0)
        s0_tup_list = [tuple(s) for s in s_init_list]
        self._snl = [np.array(s) for s in set(s0_tup_list)]

        self._train_count = 0


    def fit(self, pi_eval, stochastic):
        """
        """
        self._pi_eval = pi_eval
        self._stochastic = stochastic
        scope_name = "dsfn{}".format(self._train_count)

        dsfn = DSFN(ob_space=self._ob_space,
                   ac_space=self._ac_space,
                   horizon=self._horizon,
                   mu_dim=self._mu_dim,
                   D_train=self._D_train,
                   D_val=self._D_val,
                   phi=self._phi,
                   gamma=self._gamma,
                   hiddens=[128, 64],
                   scope_name=scope_name,
                   buffer_batch_size=64,
                   exploration_fraction=0.05,
                   exploration_initial_eps=1.0,
                   exploration_final_eps=0.01,
                   prioritized_replay=False,
                   prioritized_replay_alpha=0.6,
                   prioritized_replay_beta0=0.9,
                   prioritized_replay_beta_iters=None,
                   prioritized_replay_eps=1e-6
                   )

        self._estimator = dsfn.train(pi_eval=pi_eval)
        self._train_count += 1


    def _estimate(self, s, a, sample_size=5):
        """

        sample multiple times to reduce variance

        Returns
        -------

        numpy array

        E[\mu(s,a) | s0=s, a0=a, pi_eval]

        """
        # s = (N, d)
        #assert(len(s.shape)==2)
        # a = (N, 1)
        #assert(len(a.shape)==2)
        if self._stochastic:
            mu_est_list = []
            for _ in range(sample_size):
                mu_est_list.append(self._estimator(self._stochastic, s, a))

            mu_est = np.mean(mu_est_list, axis=0)
        else:
            mu_est = self._estimator(self._stochastic, s, a)

        return mu_est


    def estimate(self, s=None, a=None, n_sample=5):
        """TODO: Docstring for estimate.

        Parameters
        ----------
        s : TODO
        a : TODO
        n_sample : TODO, optional

        Returns
        -------
        TODO

        """

        if s is not None:
            return self._estimate(s, a, n_sample)
        else:
            # s is None
            mu_est_list = []
            for s in self._snl:
                # we don't marginalize action out (we can)
                a = self._pi_eval.act(self._stochastic, s[np.newaxis,...])[0]
                mu_est_list.append(self._estimate(s, a, n_sample))
            return np.mean(mu_est_list, axis=0)


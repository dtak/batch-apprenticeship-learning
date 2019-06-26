import os
import time
import logging
import itertools
from multiprocessing import Pool


import gym
import cvxpy as cvx
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from core.policy import LinearQ2, MixturePolicy
from model.lstd import LSPI
from model.dqn import DQNSepsis
from core.basis import GaussianKernel, RBFKernel2
from core.feature_expectation import MCMuEstimator, EmpiricalMuEstimator, LSTDMuEstimator, DeepMuEstimator

from contrib.baselines import bench, logger

import tensorflow as tf


class MaxMarginAbbeel(object):
    """
    implementation of (Abbeel & Ng 2004)

    two versions: available

    1. max-margin (stable, computationally more heavy)
    2. projection (simpler)

    """

    def __init__(self,
                 pi_init,
                 p,
                 phi,
                 mu_exp,
                 irl_precision,
                 mdp_solver,
                 mu_estimator,
                 evaluator,
                 method="max_margin",
                 slack_scale=0.01,
                 use_slack=False,
                 reward_fn=None,
                 stochastic=True,
                 D_val=None,
                 delta=0.2
                 ):
        """TODO: to be defined1.

        Parameters
        ----------
        p : int
            dimension of phi
        q : int
            dimension of psi
        phi : function
            basis function for reward
        mu_exp : target for feature expectation IRL
        mu_estimator : function
            estimate E[mu(s_0) | pi, D]
        evaluator : function
            evaluate i.t.o perf score and action matching
        irl_precision : convergence threshold
        use_slack : whether to use slack for convex optimization
        slack_scale : scaling term
        method: max_margin or projection
        """
        self._pi_init = pi_init
        self._phi = phi
        self._p = p
        self._mu_exp = mu_exp
        self._mu_estimator = mu_estimator
        self._irl_precision = irl_precision
        self._method = method
        self._evaluator = evaluator
        self._mdp_solver = mdp_solver
        self._use_slack = use_slack
        self._slack_scale = slack_scale
        self._stochastic = stochastic
        self._D_val = D_val
        self._delta = delta


    def run(self, n_iteration):
        """TODO: Docstring for something.

        Parameters
        ----------
        n_iteration : max iteration count

        Returns
        -------

        exp results
        """
        mu_exp = self._mu_exp
        mu_estimator = self._mu_estimator
        stochastic = self._stochastic

        pi_list = []
        mu_list = []
        mu_bar_list = []
        perf_list = []
        a_match_list = []
        weight_list = []
        margin_v_list = []
        margin_mu_list = []

        pi_list.append(self._pi_init)

        mu_estimator.fit(self._pi_init, stochastic)
        mu_irl = mu_estimator.estimate(s=None, a=None)

        mu_list.append(mu_irl)
        mu_bar_list.append(mu_irl)

        weight_list.append(-1.0)
        margin_v_list.append(-1.0)
        margin_mu_list.append(-1.0)


        #perf = self._evaluator.evaluate_perf(self._pi_init, stochastic)
        #perf_list.append(perf)
        a_match = self._evaluator.evaluate_a_match(self._pi_init, stochastic)
        a_match_list.append(a_match)


        for epi_i in tqdm(range(n_iteration)):

            if self._method == "max_margin":
                W, (margin_v, margin_mu, converged) = self._optimize(mu_list, pi_list)
            elif self._method == "projection":
                W, (margin_v, margin_mu, converged, mu_bar_im1) = self._optimize_projection(mu_list, mu_bar_list, pi_list)
                mu_bar_list.append(mu_bar_im1)
            else:
                raise Exception("Unknown IRL solver")

            weight_list.append(W)
            margin_v_list.append(margin_v)
            margin_mu_list.append(margin_mu)
            logging.info("margin_v: {}".format(margin_v))
            logging.info("margin_mu: {}".format(margin_mu))
            margin_hyperplane = 2 / norm(W, 2)
            logging.info("margin_hyperplane: {}".format(margin_hyperplane))

            if converged:
                logging.info("margin_mu converged after {} iterations".format(epi_i + 1))
                break


            reward_fn = self._get_reward_fn(W=W)

            pi_irl = self._mdp_solver.solve(reward_fn, return_policy=True)

            pi_list.append(pi_irl)
            #logging.debug("pi_irl: {}".format(pi_irl._W))

            mu_estimator.fit(pi_irl, stochastic)
            mu_irl = mu_estimator.estimate()

            mu_list.append(mu_irl)
            logging.debug("mu_irl: {}".format(mu_irl))

            #perf = self._evaluator.evaluate_perf(pi_irl, stochastic)
            #perf_list.append(perf)

            a_match = self._evaluator.evaluate_a_match(pi_irl, stochastic)
            a_match_list.append(a_match)

            if (1 - a_match) < self._delta:
                # validation error
                break

            mu_list_ = np.array([mu.flatten() for mu in mu_list])
            mixture_weight_list = self._choose_mixture_weight(mu_list_, self._mu_exp)

            pi_best = MixturePolicy(mixture_weight_list, pi_list)

            best_mu = mixture_weight_list.T.dot(mu_list_)
            w_best = self._mu_exp - best_mu
            w_best /= norm(w_best, 2)


        results = {
                "perf" : perf_list,
                "a_match" : a_match_list,
                "margin_v": margin_v_list,
                "margin_mu": margin_mu_list,
                "mu": mu_list,
                "weight": weight_list,
                "policy": pi_list,
                "policy_best" : pi_best,
                "weight_best" : w_best,
                }
        return results


    def _choose_mixture_weight(self, mu_list, mu_exp):
        """
        implement the choice of policy in
        Section 3.0 in Abbeel, Ng (2004)

        Parameters
        ----------
        mu_list : TODO

        Returns
        -------
        pi_best

        """
        lamda = cvx.Variable(len(mu_list))

        obj = cvx.Minimize(cvx.norm(mu_exp - (mu_list.T * lamda), 3))
        constraints = [lamda >= 0,
                      sum(lamda) == 1]


        prob = cvx.Problem(obj, constraints)
        prob.solve()

        if prob.status in ["unbounded", "infeasible"]:
            logging.warning("the optimization failed: {}".format(prob.status))

        weight_list = np.array(lamda.value).flatten()
        tol = 1e-6
        weight_list[np.abs(weight_list) < tol] = 0.0
        weight_list /= np.sum(weight_list)
        return weight_list


    def _get_reward_fn(self, W):
        """linearly parametrize reward function.

        Parameters
        ----------
        W : weight

        Returns
        -------
        TODO
        - think whether to do s, a or just s

        """
        #return lambda s : W.dot(self.phi(s))
        return lambda s, a : W.T.dot(self._phi(s, a).T)


    def _optimize(self, mu_list, pi_list):
        """linearly parametrize reward function.

        implements Eq. 11 from Abbeel

        Parameters
        ----------
        W : weight

        Returns
        -------
        TODO
        - think whether to do s, a or just s

        """
        logging.info("solving for W given mu_list")
        # define variables
        W = cvx.Variable(self._p)
        t = cvx.Variable(1)

        if self._use_slack:
            xi = cvx.Variable(1)

        mu_exp = cvx.Parameter(self._p)
        mu_exp.value = self._mu_exp.flatten()

        if self._use_slack:
            C = cvx.Parameter(1)
            C.value = self._slack_scale
            obj = cvx.Maximize(t - C * xi)
        else:
            obj = cvx.Maximize(t)

        constraints = []

        for mu in mu_list:
            mu = mu.flatten()
            if self._use_slack:
                constraints += [W.T * mu_exp + xi >= W.T * mu + t]
            else:
                constraints += [W.T * mu_exp >= W.T * mu + t]
        constraints += [cvx.norm(W, 2) <= 1]

        prob = cvx.Problem(obj, constraints)
        prob.solve()

        if prob.status in ["unbounded", "infeasible"]:
            logging.warning("the optimization failed: {}".format(prob.status))


        W = np.array(W.value)
        margin_v = t.value

        mu_list = np.array([mu.flatten() for mu in mu_list])
        margin_mu_list = norm(np.array(mu_exp.value).T - mu_list, 2, axis=1)
        margin_mu = np.min(margin_mu_list)

        converged = margin_mu <= self._irl_precision
        return W, (margin_v, margin_mu, converged)


    def _optimize_projection(self, mu_list, mu_bar_list, pi_list):
        """linearly parametrize reward function.

        implements Sec. 3.1 from Abbeel, Ng (2004)

        Parameters
        ----------
        W : weight

        Returns
        -------
        TODO
        - think whether to do s, a or just s

        """
        mu_e = self._mu_exp
        mu_im1 = mu_list[-1]
        mu_bar_im2 = mu_bar_list[-1]

        if len(mu_bar_list) == 1:
            mu_bar_im1 = mu_list[-1]
            w_i = mu_e - mu_im1
        else:
            a = mu_im1 - mu_bar_im2
            b = mu_e - mu_bar_im2
            mu_bar_im1 = (mu_bar_im2 + a.T.dot(b) / norm(a)**2) * a
            w_i = mu_e - mu_bar_im1


        w_i /= np.linalg.norm(w_i, 2)

        t_i = np.linalg.norm(w_i, 2)

        margin_v = w_i.T.dot(mu_e - mu_bar_im1)
        margin_mu = t_i

        converged = margin_mu <= self._irl_precision
        return w_i, (margin_v, margin_mu, converged, mu_bar_im1)


def train_mma(pi_0, phi_sa_dim, task_desc, params, D, evaluator, ob_space=None, ac_space=None):
    gym.logger.setLevel(logging.WARN)

    gamma =  task_desc["gamma"]
    horizon = task_desc["horizon"]
    eps = params["eps"]
    p = q = phi_sa_dim # adding action dim
    phi = D["phi_fn"]
    phi_s = D["phi_fn_s"]
    stochastic = True
    mu_estimator_type = params["mu_estimator"]
    n_action = task_desc["n_action"]
    assert isinstance(n_action, int)
    action_list = range(n_action)
    precision = params["precision"]

    mu_exp_estimator = EmpiricalMuEstimator(phi, gamma)
    mu_exp_estimator.fit(D, stochastic, return_s_init=True)
    mu_exp, s_init_list = mu_exp_estimator.estimate()


    logger.log("fitting {}".format(mu_estimator_type))
    if task_desc["type"] == "gym":
        env = gym.make(task_desc["env_id"])
        ac_space = env.action_space
        ob_space = env.observation_space
        mu_dim = p # only for discrete action
    elif task_desc["type"] == "sepsis":
        if ac_space is None:
            ac_space = (5, )
        if ob_space is None:
            ob_space = (46, )
        mu_dim = p

    stochastic = True

    s = D["s"]
    a = D["a"]
    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=1)
    s_next = D["s_next"]
    done = D["done"]
    if len(done.shape) == 1:
        done = np.expand_dims(done, axis=1)
    phi_sa = D["phi_sa"]

    n_transition = D["s"].shape[0]
    idx = idx = int(n_transition * 0.7)

    D_train = {"s" : s[:idx, :],
             "a" : a[:idx, :],
             "phi_sa" : phi_sa[:idx, :],
             "s_next": s_next[:idx, :],
             "done": done[:idx, :]}

    D_val = {"s" : s[idx:, :],
           "a" : a[idx:, :],
           "phi_sa" : phi_sa[idx:, :],
           "s_next": s_next[idx:, :],
           "done": done[idx:, :]}


    if mu_estimator_type == "lstd":
        mu_estimator = LSTDMuEstimator(phi, gamma, D, p, q, eps, s_init_list)
    elif mu_estimator_type == "dsfn":
        mu_estimator = DeepMuEstimator(phi, gamma, D_train, D_val, s_init_list, ob_space,
                ac_space, mu_dim, horizon)
    else:
        raise NotImplementedError

    if params["mdp_solver"] == "lspi":
        W_0 = np.random.normal(loc=0, scale=0.1, size=p)
        lspi = LSPI(D=D,
                    action_list=range(task_desc["n_action"]),
                    p=p,
                    gamma=gamma,
                    precision=precision,
                    lstd_eps=params["eps"],
                    W_0=W_0,
                    reward_fn=None,
                    stochastic=True,
                    max_iter=10)
        mdp_solver = lspi

    elif params["mdp_solver"] == "dqn":
        mdp_solver = DQNSepsis(D=D_train)
    else:
        raise NotImplementedError


    mma = MaxMarginAbbeel(pi_init=pi_0,
                          p=p,
                          phi=phi,
                          mu_exp=mu_exp,
                          mdp_solver=mdp_solver,
                          evaluator=evaluator,
                          irl_precision=params["precision"],
                          method=params["method"],
                          mu_estimator=mu_estimator,
                          stochastic=stochastic,
                          D_val=D_val)

    results = mma.run(n_iteration=params["n_iteration"])
    return results



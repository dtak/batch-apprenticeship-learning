"""
need to be updated
"""

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import cvxpy as cvx
import logging

from model.lstd import LSTDQ, LSTDMu, LSPI
from model.policy import LinearQ2
from model.apprenticeship_learning import BatchApprenticeshipLearning, ApprenticeshipLearning


class BatchMaxMarginPlanning(BatchApprenticeshipLearning):
    """Docstring for BatchMaxMarginPlanning. """

    def __init__(self,
                 pi_init,
                 D,
                 p,
                 q,
                 action_list,
                 gamma,
                 lstd_eps,
                 mu_exp,
                 s_init_sampler,
                 s_init_list,
                 mu_sample_size,
                 irl_precision,
                 slack_scale,
                 mdp_solver_name,
                 use_slack=True,
                 reward_fn=None,
                 max_iter=3,
                 alpha=0.1
            ):
        super().__init__(
                         pi_init=pi_init,
                         D=D,
                         p=p,
                         q=q,
                         action_list=action_list,
                         gamma=gamma,
                         lstd_eps=lstd_eps,
                         mu_exp=mu_exp,
                         s_init_sampler=s_init_sampler,
                         s_init_list=s_init_list,
                         mu_sample_size=mu_sample_size,
                         irl_precision=irl_precision,
                         slack_scale=slack_scale,
                         mdp_solver_name=mdp_solver_name,
                         use_slack=use_slack,
                         reward_fn=reward_fn,
                         max_iter=max_iter)

        self._alpha = alpha
        self._slack_q = 1
        self._loss_list = []


    def _optimize(self, mu_list, pi_list):
        """linearly parametrize reward function.

        implements eq 11 from Abbeel
        note: we can rewrite this as an SVM problem

        Parameters
        ----------
        W : weight

        Returns
        -------
        - think whether to do s, a or just s

        """
        logging.info("solving for W given mu_list")
        # define variables
        W = cvx.Variable(self._p)
        t = cvx.Variable(1)

        if self._use_slack:
            #xi = cvx.Variable(len(mu_list))
            xi = cvx.Variable(1)

        mu_exp = cvx.Parameter(self._p)
        mu_exp.value = self._mu_exp.flatten()

        C = cvx.Parameter(sign='Positive', value=self._slack_scale)

        # since obj is max
        # we should penalize xi with minus
        # bc. xi bumps up expert's reward (see below)
        obj = cvx.Minimize(0.5 * cvx.square(cvx.norm(W, 2)) + C * cvx.norm(xi, self._slack_q))

        constraints = []


        l = self._loss(pi_list[-1])
        logging.info("loss for pi: {}".format(l))
        self._loss_list.append(l)

        for mu, l in zip(mu_list, self._loss_list):
            mu = mu.flatten()
            constraints += [W.T * mu_exp + xi >= W.T * mu + self._alpha * l]
        constraints += [cvx.norm(W, 2) <= 1]

        prob = cvx.Problem(obj, constraints)
        prob.solve()

        print("slack", xi.value)

        if prob.status in ["unbounded", "infeasible"]:
            logging.warning("the optimization failed: {}".format(prob.status))

        # if svm formulation, need to normalize
        # W = W.value / np.linalg(W.value, 2)
        W = np.array(W.value)
        v_list = np.array([W.T.dot(mu.flatten()) for mu in mu_list])
        v_exp = np.array(W.T.dot(mu_exp.value)).flatten()
        v_margin_list = norm(v_exp - v_list, 2, axis=1)
        margin_v = np.min(v_margin_list)

        # convergence in mu implies convergence in value (induced convergence)
        # but we don't use this relation here
        mu_list = np.array([mu.flatten() for mu in mu_list])
        margin_mu_list = norm(np.array(mu_exp.value).T - mu_list, 2, axis=1)
        margin_mu = np.min(margin_mu_list)

        converged = margin_mu <= self._irl_precision

        return W, (margin_v, margin_mu, converged)


    def _loss(self, pi):
        """TODO: Docstring for loss.

        Parameters
        ----------
        pi : TODO

        what is the rate of non-matching?

        Returns
        -------
        TODO

        """

        a = np.vstack([pi.choose_action(s) for s in self._D["s"]])
        return np.mean(a != self._D["a"])



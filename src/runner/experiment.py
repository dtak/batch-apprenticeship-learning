import time

from algo.apprenticeship_learning import BatchApprenticeshipLearning as BAL
from algo.max_margin_planning import BatchMaxMarginPlanning as BMMP

from algo.fa import LinearQ3, Estimator
from algo.lstd import LSTDQ, LSTDMu, LSPI
from algo.policy import RandomPolicy2
from algo.dqn import DQN

import util.plotting


class Experiment(object):
    """Docstring for Experiment. """

    def __init__(self, args):
        """TODO: to be defined1.

        Parameters
        ----------
        pi_init : TODO
        eid : TODO

        """
        if args.eid == "":
            self._eid = "no_eid"
        else:
            self._eid = args.eid

    def run(self, algo):
        """TODO: Docstring for run.

        Parameters
        ----------
        algo : TODO

        Returns
        -------
        TODO

        """
        mgr = self.manager

        # load batch data here

        if algo == 'al' or algo == 'apprenticeship_learning':
            bal = BAL(
                      pi_init=mgr._pi_init,
                      D=mgr._D,
                      # do we need this?
                      action_list=mgr._action_list,
                      p=mgr._p, # basis
                      q=mgr._q, # lstd-mu
                      gamma=mgr._gamma,
                      irl_precision=mgr._irl_precision,
                      lstd_eps=mgr._lstd_eps,
                      mu_exp=mgr._mu_exp,
                      s_init_sampler=mgr._s_init_sampler,
                      s_init_list=mgr._s_init_list,
                      mu_sample_size=mgr._mu_sample_size,
                      mdp_solver_name=mgr._mdp_solver_name,
                      use_slack=mgr._use_slack,
                      slack_scale=mgr._slack_scale
                      )
            results = bal.run(n_trial=mgr._n_trial, n_iteration=mgr._n_iteration)

        elif algo == 'mmp' or algo == 'maximum_margin_planning':
            bmmp = BMMP(
                      pi_init=mgr._pi_init,
                      D=mgr._D,
                      # do we need this?
                      action_list=mgr._action_list,
                      p=mgr._p, # basis
                      q=mgr._q, # lstd-mu
                      gamma=mgr._gamma,
                      irl_precision=mgr._irl_precision,
                      lstd_eps=mgr._lstd_eps,
                      mu_exp=mgr._mu_exp,
                      s_init_sampler=mgr._s_init_sampler,
                      s_init_list=mgr._s_init_list,
                      mu_sample_size=mgr._mu_sample_size,
                      mdp_solver_name=mgr._mdp_solver_name,
                      use_slack=True,
                      slack_scale=mgr._slack_scale,
                      alpha=mgr._alpha
                      )
            results = bmmp.run(n_trial=mgr._n_trial, n_iteration=mgr._n_iteration)
        else:
            raise NotImplementedError
        return results


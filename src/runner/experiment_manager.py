from multiprocessing import Pool
import pickle
import numpy as np
import pandas as pd
import time
import logging
import os
notebook_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(notebook_path, os.pardir))

from algo.lstd import LSTDQ, LSTDMu, LSPI
from algo.policy import *

from util.basis import BasisFunction

from experiment.experiment import Experiment


DATA_PATH = "data/D_train.pkl"
DATA_PCA_PATH = "data/D_train_pca.pkl"
DATA_COMPACT_PATH = "data/D_train_compact.pkl"
DATA_SIGMOID_COMPACT_PATH = "data/D_train_sigmoid_compact.pkl"
DATA_SIGMOID_PATH = "data/D_train_sigmoid.pkl"
KNN_POLICY_PATH = "data/knn_policy.pkl"


class ExperimentManager():
    def __init__(self, args):
        '''
        experiments: a list of Experiment instances
        '''
        # define hyperparameters

        # MDP
        self._gamma = args.gamma
        self._n_trial = args.n_trial
        self._n_iteration = args.n_iteration
        self._mdp_solver_name = args.mdp_solver_name

        # lstd

        self._lstd_eps = args.lstd_eps

        # irl
        self._algo = args.algo # mmp or al
        self._use_slack = args.use_slack
        self._slack_scale = args.slack_scale
        self._alpha = args.alpha # scale of loss
        self._irl_precision = args.irl_precision
        self._mu_sample_size = args.mu_sample_size # depends on stochasticity

        # data processing
        self._use_pca = args.use_pca
        self._feature_type = args.feature_type

        # etc
        self._parallelized = args.parallelized

        # register mdp info
        self._env = args.env
        if self._env == "sepsis":
            action_list = list(range(25))
        elif self._env == "MountainCar-v0":
            action_list = list(range(3))
        else:
            raise Exception("Unknown Env")
        self._action_list = action_list
        self._n_action = len(action_list)

        # construct D_phi
        self._load_data()

        # construct basis features


        # experiments
        self.experiments = []


    def _load_data(self):
        """TODO: Docstring for _load_data.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
        if self._feature_type == "pca":
            path = os.path.join(root_path, DATA_PCA_PATH)
        elif self._feature_type == "compact":
            path = os.path.join(root_path, DATA_COMPACT_PATH)
        elif self._feature_type == "sigmoid_compact":
            path = os.path.join(root_path, DATA_SIGMOID_COMPACT_PATH)
        elif self._feature_type == "sigmoid":
            path = os.path.join(root_path, DATA_SIGMOID_PATH)
        else:
            path = os.path.join(root_path, DATA_PATH)

        with open(path, "rb") as f:
            D = pickle.load(f)
        self._D_s_features = D["features"]
        # n x k
        s = D["s"].astype(np.float)
        # n x 1
        a = np.vstack(D["a"]).astype(np.int)
        # n x 1
        r = np.vstack(D["r"]).astype(np.float)
        # n x k
        s_next = D["s_next"]
        s_next[s_next == None] = 0.
        s_next[np.isnan(s_next)] = 0.
        s_next = s_next.astype(np.float)
        assert np.sum(s_next == None) == 0
        assert np.sum(np.isnan(s_next)) == 0
        # n x 1
        done = np.vstack(D["done"]).astype(np.int)
        # n x 1

        # n x (k + 1+ 1 + k + 1)
        #D_mat = np.hstack((s, a, r, s_next, done))
        #self._D = D_mat

        # n x p (=ka)
        phi_s = D["phi_s"]
        phi_s_next = D["phi_s_next"]


        if self._feature_type in ["sigmoid", "sigmoid_compact"]:
            phi_criteria = None
        phi_criteria = np.array(D["phi_criteria"]).astype(np.float)


        phi_s_next[phi_s_next == None] = 0
        phi_s_next = phi_s_next.astype(np.int)
        phi_s_next[np.isnan(phi_s_next)] = 0
        assert np.sum(phi_s_next == None) == 0
        assert np.sum(np.isnan(phi_s_next)) == 0

        self._p = phi_s.shape[1] * self._n_action
        self._q = phi_s.shape[1] * self._n_action
        bf = BasisFunction(self._p,
                           self._n_action,
                           phi_s,
                           phi_s_next,
                           phi_criteria)

        self._D = {}
        self._D["s"] = s
        self._D["a"] = a
        self._D["r"] = r
        self._D["s_next"] = s_next
        self._D["done"] = done
        self._D["phi_sa"] = bf.phi_sa(a)
        self._D["phi_sa_next_fn"] = bf.phi_sa_next
        self._D["phi_s_next"] = phi_s_next
        self._D["phi_fn"] = bf.transform



        # get empirical initial states
        self._s_init_list = D["s_init"].astype(np.float)

        def get_sampler(s_init_list):
            n = s_init_list.shape[0]
            def sample():
                idx = np.random.choice(range(n))
                return s_init_list[idx, :]
            return sample

        self._s_init_sampler = get_sampler(self._s_init_list)

        # brew initial policy for irl
        #self._pi_init = RandomPolicy2(self._action_list)
        #self._pi_init = RandomPolicy2([0, 1, 2, 3, 4])

        #p = [0.252003,
        #     0.173701,
        #     0.133273,
        #     0.148427,
        #     0.123073,
        #     0.003648,
        #     0.008363,
        #     0.011626,
        #     0.008257,
        #     0.010545,
        #     0.002918,
        #     0.007249,
        #     0.009258,
        #     0.008907,
        #     0.014126,
        #     0.002653,
        #     0.006187,
        #     0.008860,
        #     0.008860,
        #     0.015824,
        #     0.002719,
        #     0.004079,
        #     0.005657,
        #     0.009338,
        #     0.020453]
        #p = np.array(p)
        #p /= p.sum()

        self._pi_init = StochasticPolicy2(p=p)
        #self._pi_init = KNNPolicy(s=s, a=a, k=20, path=KNN_POLICY_PATH)


        # load precomputed mu_exp
        self._mu_exp = np.array(D["mu_exp"][self._gamma].tolist())


    def save_experiment(self, res, exp):
        eid = exp._eid
        path = os.path.join(root_path, "data/res_irl_{}_{}.pkl".format(eid, time.time()))
        with open(path, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


    def set_experiment(self, exp):
        exp.manager = self
        self.experiments.append(exp)


    def _run(self, exp):
        res = exp.run(algo=self._algo)
        self.save_experiment(res, exp)

    def run(self):
        '''
        parallelize the experiments
        '''
        if self._parallelized:
            try:
                pool = Pool(os.cpu_count() - 1)
                pool.map(self._run, self.experiments)
            finally:
                pool.close()
                pool.join()
        else:
            for e in self.experiments:
                self._run(e)




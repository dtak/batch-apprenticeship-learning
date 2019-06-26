import numpy as np
from numpy.linalg import norm
import torch

from collections import Counter
import pickle
import os
import logging


class EpsilonGreedyPolicy:
    '''
    TODO: refactor this
    '''
    def __init__(self, num_states, num_actions, epsilon, Q=None):
        if Q is None:
            self._Q = np.zeros((num_states, num_actions))
        else:
            self._Q = Q
        self._eps = epsilon

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)

    def query_Q_probs(self, s=None, a=None):
        Q_probs = np.zeros(self._Q.shape)
        for s in range(self._Q.shape[0]):
            Q_probs[s, :] = self.query_Q_probs(s)
        if s is None and a is None:
            return Q_probs
        elif a is None:
            return Q_probs[s, :]
        else:
            return Q_probs[s, a]


    def _query_Q_probs(self, s, a=None):
        num_actions = self._Q.shape[1]
        probs = np.ones(num_actions, dtype=float) * self._eps / num_actions
        ties = np.flatnonzero(self._Q[s, :] == self._Q[s, :].max())
        if a is None:
            best_a = np.random.choice(ties)
            probs[best_a] += 1. - self._eps
            return probs
        else:
            if a in ties:
                probs[a] += 1. - self._eps
            return probs[a]

    def choose_action(self, s):
        probs = self._query_Q_probs(s)
        return np.random.choice(len(probs), p=probs)

    def update_Q_val(self, s, a, val):
        self._Q[s,a] = val


class GreedyPolicy:
    def __init__(self, num_states, num_actions, Q=None):
        if Q is None:
            # start with random policy
            self._Q = np.zeros((num_states, num_actions))
        else:
            # in case we want to import e-greedy
            self._Q = Q

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)

    def query_Q_probs(self, s=None, a=None):
        Q_probs = np.zeros(self._Q.shape).astype(np.float)
        for s in range(self._Q.shape[0]):
            ties = np.flatnonzero(self._Q[s, :] == self._Q[s, :].max())
            a = np.random.choice(ties)
            Q_probs[s, a] = 1.0
        if s is None and a is None:
            return Q_probs
        elif a is None:
            return Q_probs[s, :]
        else:
            return Q_probs[s, a]

    def choose_action(self, s):
        ties = np.flatnonzero(self._Q[s, :] == self._Q[s, :].max())
        return np.random.choice(ties)

    def get_opt_actions(self):
        opt_actions = np.zeros(self._Q.shape[0])
        for s in range(opt_actions.shape[0]):
            opt_actions[s] = self.choose_action(s)
        return opt_actions

    def update_Q_val(self, s, a, val):
        self._Q[s,a] = val


class StochasticPolicy:
    def __init__(self, num_states, num_actions, Q=None):
        if Q is None:
            # start with random policy
            self._Q = np.zeros((num_states, num_actions))
        else:
            # in case we want to import e-greedy
            # make Q non negative to be useful as probs
            self._Q = Q

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)

    def query_Q_probs(self, s=None, a=None, laplacian_smoothing=True):
        '''
        returns:
            probability distribution of actions over all states
        '''
        if laplacian_smoothing:
            LAPLACIAN_SMOOTHER = 0.01
            L = (np.max(self._Q, axis=1) - np.min(self._Q, axis=1))* LAPLACIAN_SMOOTHER
            Q = self._Q - np.expand_dims(np.min(self._Q, axis=1) - L, axis=1)
        else:
            Q = self._Q - np.expand_dims(np.min(self._Q, axis=1), axis=1)
        Q_sum = np.sum(Q, axis=1)
        # if zero, we give uniform probs with some gaussian noise
        num_actions = self._Q.shape[1]
        Q[Q_sum==0, :] = 1.
        Q_sum[Q_sum==0] = num_actions
        Q_probs = Q / np.expand_dims(Q_sum, axis=1)
        Q_probs[Q_sum==0, :] += np.random.normal(0, 1e-4, num_actions)

        if s is None and a is None:
            return Q_probs
        elif a is None:
            return Q_probs[s, :]
        else:
            return Q_probs[s, a]


    def choose_action(self, s, laplacian_smoothing=True):
        probs = self.query_Q_probs(s, laplacian_smoothing=laplacian_smoothing)
        return np.random.choice(len(probs), p=probs)

    def update_Q_val(self, s, a, val):
        self._Q[s,a] = val


class RandomPolicy:
    def __init__(self, num_states, num_actions):
        self._Q_probs = np.ones((num_states, num_actions), dtype=float) / num_actions

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q_probs)

    def choose_action(self, s):
        probs = self._Q_probs[s, :]
        return np.random.choice(len(probs), p=probs)


class RandomPolicy2:
    def __init__(self, choices):
        self._choices = choices

    def choose_action(self, s):
        """ sample uniformly """
        return np.random.choice(self._choices)


class StochasticPolicy2:
    def __init__(self, p):
        self._p = p

    def choose_action(self, s):
        return np.random.choice(len(self._p), p=self._p)


class EmpiricalPolicy:
    """
    input: D
    output .act(s)

    logic:

    if s in D["s"]

    """
    def __init__(self, D, n_action, n_neighbors=3):
        self._D = D
        self._n_action = n_action

        X = D["s"]
        y = D["a"]

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, y)
        self._knn = knn

    def act(self, stochastic, s):
        if stochastic:
            probs = self._knn.predict_proba(s)[0]
            return (np.random.choice(len(probs), p=probs), s)
        else:
            return (knn.predict(s), s)


class KNNPolicy(object):
    """Docstring for KNNPolicy. """

    def __init__(self, s, a, k=20, path=None):
        """TODO: to be defined1.

        Parameters
        ----------
        s : TODO
        a : TODO
        k : TODO, optional


        """
        self._s = s
        self._a = a
        self._n = a.shape[0]

        if os.path.exists(path):
            with open(path, "rb") as f:
                res = pickle.load(f)
            self._knn, k_saved = res

            if k != k_saved:
                self._knn = self.generate_knn(k, path)
        else:
            self._knn = self.generate_knn(k, path)


    def generate_knn(self, k, path):
        """TODO: Docstring for generate_knn.

        Parameters
        ----------
        k : TODO
        path : TODO

        Returns
        -------
        TODO

        """
        adj_mat = {}
        knn = {}
        logging.info("creating knn policy from scratch")
        for i in range(self._n):
            s_i = self._s[i]
            for j in range(self._n):
                s_j = self._s[j]
                if tuple(s_i) not in adj_mat:
                    adj_mat[tuple(s_i)] = []
                t = (np.asscalar(self._a[i]), norm(s_i - s_j))
                adj_mat[tuple(s_i)].append(t)


            dist = adj_mat[tuple(s_i)]
            k_neighbors = sorted(dist, key=lambda x : x[1])[:k]
            c = Counter([n[0] for n in k_neighbors]).most_common(1)[0][0]
            knn[tuple(s_i)] = c

        res = (knn, k)

        with open(path, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

        return knn

    def choose_action(self, s):
        """TODO: Docstring for choose_action.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
        return self._knn[tuple(s)]


class MixturePolicy(object):
    """Docstring for MixturePolicy. """

    def __init__(self, weight_list, pi_list):
        """TODO: to be defined1.

        Parameters
        ----------
        weights : mixture weights (should have been normalized)
        pi_list : policies


        """
        if not np.isclose(np.sum(weight_list), 1.0):
            raise Exception("normalize mixtures weights to sum to 1.0")
        self._weight_list = weight_list
        self._pi_list = pi_list


    def choose_action(self, s):
        """TODO: Docstring for choose_action.

        Parameters
        ----------
        s : state

        Returns
        -------
        TODO

        """
        i = np.random.choice(range(len(self._pi_list)), p=self._weight_list)
        return self._pi_list[i].choose_action(s)


import torch
from torch import nn, optim
from torch.autograd import Variable


class LinearQ(nn.Module):
    """Docstring for LinearQ. """

    def __init__(self, phi, k):
        """TODO: to be defined1.

        Parameters
        ----------
        phi : basis function for (s, a)
        k : feature dimension
        """
        super().__init__()
        self._phi = phi
        self._l1 = nn.Linear(k, 1)


    def forward(self, s, a):
        """
        predict Q(s,a)
        """
        x = self._phi(s, a)
        out = self._l1(x)
        return out


    def choose_action(self, s):
        """
        argmax_a Q(s, a)
        """
        Q_hat = np.array([self.forward(self._phi(s, a)) for a in range(n_actions)])
        ties = np.flatnonzero(Q_hat == Q_hat.max())
        return np.random.choice(ties)


class LinearQ2(object):
    """LinearQ for continuous-state, discrete_action """

    def __init__(self, action_list, phi, W):
        """TODO: to be defined1.

        Parameters
        ----------
        action_list : list of valid actions (assuming discrete)
        phi : basis function of (s, a)
        W : TODO, optional


        """
        self._action_list = action_list
        self._phi = phi
        self._W = W


    def predict(self, s, a=None):
        """TODO: Docstring for predict.

        only works for discrete action space

        Parameters
        ----------
        s : state

        Returns
        -------
        Q(s, a)

        """
        if a is None:
            q_list = []
            for a in self._action_list:
                q = np.asscalar(self._W.T.dot(self._phi(s, a).T))
                q_list.append(q)
            return np.array(q_list)
        else:
            return self._W.T.dot(self._phi(s, a))


    def choose_action(self, s):
        Q_hat = self.predict(s)
        ties = np.flatnonzero(Q_hat == Q_hat.max())
        return np.random.choice(ties)


    def act(self, stochastic, s):
        Q_hat = self.predict(s)
        ties = np.flatnonzero(Q_hat == Q_hat.max())
        return np.random.choice(ties), None


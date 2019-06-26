"""
deep sucessor feature network

check also the sript called feature_expectation.py
"""

import os
import sys
path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(path, os.pardir))
import itertools
import logging

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import contrib.baselines
from contrib.baselines import logger
from contrib.baselines import deepq
from contrib.baselines.deepq.utils import BatchInput
from contrib.baselines.common.schedules import LinearSchedule
import contrib.baselines.common.tf_util as U


class DeepSucessorFeatureNetwork(object):
    """
    DSRN

    estimates feature expectation

    """
    def __init__(self, ob_space,
                       ac_space,
                       horizon,
                       D_train,
                       D_val,
                       phi,
                       mu_dim,
                       gamma,
                       scope_name,
                       env=None,
                       model=None,
                       reuse=True,
                       hiddens=[128, 128],
                       learning_rate=1e-3,
                       max_timesteps=10**6,
                       print_freq=10,
                       layer_norm=True,
                       exploration_fraction=0.1,
                       exploration_initial_eps=1.0,
                       exploration_final_eps=0.1,
                       target_network_update_freq=500,
                       evaluation_freq=5000,
                       param_noise=False,
                       grad_norm_clipping=10,
                       buffer_batch_size=32,
                       prioritized_replay=True,
                       prioritized_replay_alpha=0.6,
                       prioritized_replay_beta0=0.9,
                       prioritized_replay_beta_iters=None,
                       prioritized_replay_eps=1e-6,
                       action_list=[],
                       delta=10.0
                       ):
        #self._model = mu_mlp(hiddens, layer_norm=layer_norm)
        #if isinstance(ac_space, spaces.Box):
        #    assert len(ac_space.shape) == 1
        #    input_space = (ob_space.shape[0] + 1, )
        #    raise NotImplementedError
        #elif isinstance(ac_space, spaces.Discrete):
        #    input_space = (ob_space.shape[0] + ac_space.n, )
        #    input_space = spaces.Box(low=-1.0, high=1.0, shape=input_space)
        #elif isinstance(ac_space, spaces.MultiDiscrete):
        #    input_space = (ob_space.shape[0] + ac_space.nvec)
        #    raise NotImplementedError
        #elif isinstance(ac_space, spaces.MultiBinary):
        #    raise NotImplementedError
        #else:
        #    raise NotImplementedError

        from gym import spaces
        mu_space = (mu_dim, )
        # hard-coded based on horizon
        mu_space = spaces.Box(low=-float(horizon), high=float(horizon), shape=mu_space)

        self._ob_space = ob_space
        self._ac_space = ac_space
        self._mu_space = mu_space
        self._hiddens = hiddens

        self._model = mu_mlp_gaussian(ob_space=ob_space,
                                      ac_space=ac_space,
                                      mu_space=mu_space,
                                      hiddens=hiddens,
                                      scope=scope_name,
                                      reuse=reuse)

        self._obs_shape = ob_space.shape

        if isinstance(ac_space, gym.spaces.Box):
            assert len(ac_space.shape) == 1
            self._acs_shape = ac_space.shape[0]
        elif isinstance(ac_space, gym.spaces.Discrete):
            self._acs_shape = (1,)

        self._n_action = ac_space.n
        self._D_train = D_train
        self._D_val = D_val
        self._mu_dim = mu_dim
        self._phi = phi

        self._scope_name = scope_name
        self._reuse = reuse

        self._hiddens = hiddens
        self._lr = learning_rate
        self._gamma = gamma
        self._max_timesteps = max_timesteps
        self._print_freq = print_freq
        self._exploration_fraction = exploration_fraction
        self._exploration_initial_eps = exploration_initial_eps
        self._exploration_final_eps = exploration_final_eps
        self._param_noise = param_noise
        self._layer_norm = layer_norm
        self._grad_norm_clipping = grad_norm_clipping
        self._buffer_batch_size = buffer_batch_size
        self._target_network_update_freq = target_network_update_freq
        self._evaluation_freq = evaluation_freq

        self._prioritized_replay = prioritized_replay
        self._prioritized_replay_alpha = prioritized_replay_alpha
        self._prioritized_replay_beta0 = prioritized_replay_beta0
        self._prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self._prioritized_replay_eps = prioritized_replay_eps

        self._delta = delta



        #s = D["s"]
        #a = D["a"]
        #if len(a.shape) == 1:
        #    a = np.expand_dims(a, axis=1)
        #s_next = D["s_next"]
        #done = D["done"]
        #if len(done.shape) == 1:
        #    done = np.expand_dims(done, axis=1)
        #phi_sa = D["phi_sa"]



        #self._n_transition = D["s"].shape[0]
        #self._idx = idx = int(self._n_transition * 0.7)

        #self._D_train = zip(s[:idx, :],
        #           a[:idx, :],
        #           phi_sa[:idx, :],
        #           s_next[:idx, :],
        #           done[:idx, :])


        #self._D_val = {"s" : s[idx:, :],
        #               "a" : a[idx:, :],
        #               "phi_sa" : phi_sa[idx:, :],
        #               "s_next": s_next[idx:, :],
        #               "done": done[idx:, :]}


        self._n_train = D_train["s"].shape[0]




    def train(self, pi_eval, stochastic=True):
        """TODO: Docstring for train.

        Parameters
        ----------
        pi_eval: function
            policy to evaluate

        Returns
        -------

        """
        self._pi = pi_eval
        self._mu_stochastic = stochastic
        with tf.Graph().as_default():
            return self._train()
        #return self._train()


    def _train(self):

        self._buffer_list = []
        self._beta_schedule_list = []
        if self._prioritized_replay:
            self._rb = PrioritizedReplayBufferNextAction(self._n_train, alpha=self._prioritized_replay_alpha)
            if self._prioritized_replay_beta_iters is None:
                self._prioritized_replay_beta_iters = self._max_timesteps
            self._bs = LinearSchedule(self._prioritized_replay_beta_iters,
                               initial_p=self._prioritized_replay_beta0,
                               final_p=1.0)
        else:
            self._rb = ReplayBufferNextAction(self._n_train)


        D_train_zipped = zip(self._D_train["s"],
                         self._D_train["a"],
                         self._D_train["phi_sa"],
                         self._D_train["s_next"],
                         self._D_train["done"])
        for (s, a, phi_sa, s_next, done) in D_train_zipped:

            a_next = self._pi.act(self._mu_stochastic, s_next[np.newaxis, ...])[0]
            self._rb.add(s, a, phi_sa.flatten(), s_next, a_next, float(done))

        phi_sa_val = self._D_val["phi_sa"]
        s_val = self._D_val["s"]
        a_val = self._D_val["a"]
        s_next_val = self._D_val["s_next"]

        a_next_val = self._pi.act(self._mu_stochastic, s_next_val)[0]
        a_next_val = a_next_val[..., np.newaxis]


        sess = tf.Session()
        sess.__enter__()

        def make_obs_ph(name):
            return BatchInput(self._obs_shape, name=name)

        def make_acs_ph(name):
            return BatchInput(self._acs_shape, name=name)


        tools = build_train(
            make_obs_ph=make_obs_ph,
            make_acs_ph=make_acs_ph,
            optimizer=tf.train.AdamOptimizer(learning_rate=self._lr),
            mu_func=self._model,
            phi_sa_dim=self._mu_dim,
            grad_norm_clipping=self._grad_norm_clipping,
            gamma=self._gamma,
            scope=self._scope_name,
            reuse=True
        )

        mu_estimator, train, update_target = tools


        self._timestep = int(self._exploration_fraction * self._max_timesteps),

        U.initialize()
        update_target()

        for t in itertools.count():
            if self._prioritized_replay:
                experience = self._rb.sample(self._buffer_batch_size,
                                             beta=self._bs.value(t + 1))
                (s, a, phi_sa, s_next, a_next, dones, weights, batch_idxes) = experience
            else:
                s, a, phi_sa, s_next, a_next, dones = self._rb.sample(self._buffer_batch_size)
                weights, batch_idxes = np.ones(self._buffer_batch_size), None


            if len(a_next.shape) == 1:
                a_next = np.expand_dims(a_next, axis=1)


            td_errors = train(self._mu_stochastic, s, a, phi_sa, s_next, a_next, dones, weights)

            if self._prioritized_replay:
                new_priorities = np.abs(td_errors) + self._prioritized_replay_eps
                self._rb.update_priorities(batch_idxes, new_priorities)


            if t % self._target_network_update_freq == 0:
                #sys.stdout.flush()
                #sys.stdout.write("average training td_errors: {}".format(td_errors.mean()))
                logger.log("average training td_errors: {}".format(td_errors.mean()))
                update_target()


            if t % self._evaluation_freq == 0:
                logger.log("been trained {} steps".format(t))


                mu_est_val = mu_estimator(self._mu_stochastic, s_val, a_val)
                mu_target_val = phi_sa_val + self._gamma * mu_estimator(self._mu_stochastic, s_next_val, a_next_val)
                # average over rows and cols
                td_errors_val = np.mean((mu_est_val - mu_target_val)**2)


                if td_errors_val < self._delta:
                    logger.log("mean validation td_errors: {}".format(td_errors_val))
                    break


            if t > self._max_timesteps:
                break

        self._mu_estimator = mu_estimator
        return mu_estimator


    def predict(self, s, a, stochastic=True):
        mu_est = self._mu_estimator(s[None], a[None], stochastic)
        return mu_est[0]


def build_train(make_obs_ph, make_acs_ph, optimizer, mu_func,
                phi_sa_dim, scope, reuse, grad_norm_clipping=None,
                gamma=1.0, double_q=True, mu_stochastic=True,
                param_noise=False, param_noise_filter_func=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        obs_t_input = make_obs_ph("obs_t")
        act_t_input = make_acs_ph("act_t")
        phi_sa_t_ph = tf.placeholder(tf.float32, [None, phi_sa_dim],  name="phi_sa")
        obs_tp1_input = make_obs_ph("obs_tp1")
        act_tp1_input = make_acs_ph("act_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")
        mu_stochastic_ph = U.get_placeholder(name="mu_stochastic", dtype=tf.bool, shape=())

        mu_t_est = mu_func(mu_stochastic_ph, obs_t_input.get(), act_t_input.get())

        mu_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        scope=tf.get_variable_scope().name)

        mu_tp1_est = mu_func(mu_stochastic_ph, obs_tp1_input.get(), act_tp1_input.get())

        target_mu_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        scope=tf.get_variable_scope().name)

        mask = tf.expand_dims(1.0 - done_mask_ph, axis=1)
        mu_tp1_est_masked = tf.multiply(mu_tp1_est, mask)

        mu_t_target = phi_sa_t_ph + gamma * mu_tp1_est_masked


        td_error = mu_t_est - tf.stop_gradient(mu_t_target)
        td_error = tf.reduce_sum(tf.square(td_error), 1)

        errors = td_error
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_error, var_list=mu_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=mu_func_vars)

        update_target_expr = []
        for var, var_target in zip(sorted(mu_func_vars, key=lambda v: v.name),
                                   sorted(target_mu_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        train = U.function(
            inputs=[
                mu_stochastic_ph,
                obs_t_input,
                act_t_input,
                phi_sa_t_ph,
                obs_tp1_input,
                act_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )

        update_target = U.function([], [], updates=[update_target_expr])

        mu_estimator = U.function([mu_stochastic_ph, obs_t_input, act_t_input], mu_t_est)

        return mu_estimator, train, update_target


def _mlp(hiddens, inpt, phi_sa_dim, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.tanh(out)
        mu_out = layers.fully_connected(out, num_outputs=phi_sa_dim, activation_fn=None)
        return mu_out


def mu_mlp(hiddens=[], layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)


from contrib.baselines.common.mpi_running_mean_std import RunningMeanStd
from contrib.baselines.common.distributions import make_pdtype
from contrib.baselines.acktr.utils import dense

def _mu_mlp_gaussian(hiddens, scope, reuse,
        ob_space, ac_space, mu_space,
        mu_stochastic, ob, ac):

    num_hid_layers = 2
    hid_size = hiddens[0]
    assert isinstance(ob_space, gym.spaces.Box)

    ac_pdtype = make_pdtype(ac_space)
    sequence_length = None


    with tf.variable_scope("obfilter", reuse=tf.AUTO_REUSE):
        ob_rms = RunningMeanStd(shape=ob_space.shape)

    obz = tf.clip_by_value((ob - ob_rms.mean) / ob_rms.std, -5.0, 5.0)

    mu_pdtype = make_pdtype(mu_space)

    #if isinstance(ac_space, gym.spaces.Box):
    #    # if continuous action
    #    mu_ac = U.get_placeholder(name="act_t", dtype=tf.float32,
    #            shape=[sequence_length] + [ac_pdtype.param_shape()[0]//2])
    #else:
    #    # if discrete action
    #    mu_ac = U.get_placeholder(name="act_t", dtype=tf.float32,
    #            shape=[sequence_length] + list(ac_space.shape))

    ob_ac_input = tf.concat([ob, ac], axis=1, name="mu_input")

    last_out = ob_ac_input

    for i in range(num_hid_layers):
        last_out = tf.nn.tanh(dense(last_out, hid_size, "mu%i" % (i+1),
            weight_init=U.normc_initializer(1.0), reuse=tf.AUTO_REUSE))

    mu_mean = dense(ob_ac_input, mu_space.shape[0], "mu_final_mean",
            U.normc_initializer(0.01), reuse=tf.AUTO_REUSE)

    mu_logstd = tf.get_variable(name="mu_final_logstd", shape=[1,
        mu_space.shape[0]], initializer=tf.zeros_initializer())

    mu_pdparam = tf.concat([mu_mean, mu_mean * 0.0 + mu_logstd], axis=1)

    mu = mu_pdtype.pdfromflat(mu_pdparam)


    mu_est = U.switch(mu_stochastic, mu.sample(), mu.mode())
    #mu_func = U.function([mu_stochastic, ob, ac], [mu_est])

    return mu_est


def mu_mlp_gaussian(hiddens, scope, reuse, ob_space, ac_space, mu_space):
    return lambda *args, **kwargs: _mu_mlp_gaussian(hiddens, scope, reuse,
                                                    ob_space, ac_space,
                                                    mu_space, *args, **kwargs)


#def mu_mlp_gaussian(ob_space, ac_space, mu_space, hiddens, reuse=False):
#    return MlpPolicyMultiGaussian(ob_space=ob_space,
#                                  ac_space=ac_space,
#                                  mu_space=mu_space,
#                                  hid_size=hiddens[0],
#                                  num_hid_layers=2)


class ReplayBufferNextAction(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, act, reward, obs_tp1, act_tp1, done):
        data = (obs_t, act, reward, obs_tp1, act_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, acs_t, rewards, obses_tp1, acs_tp1, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, act_t, reward, obs_tp1, act_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            acs_t.append(np.array(act_t, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            acs_tp1.append(np.array(act_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(acs_t), np.array(rewards), np.array(obses_tp1), np.array(acs_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
import random
class PrioritizedReplayBufferNextAction(ReplayBufferNextAction):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBufferNextAction, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

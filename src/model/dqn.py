import itertools
import numpy as np
import tensorflow as tf

import contrib.baselines
from contrib.baselines import logger
from contrib.baselines import deepq
from contrib.baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from contrib.baselines.deepq.utils import BatchInput
from contrib.baselines.common.schedules import LinearSchedule
import contrib.baselines.common.tf_util as U


import os
path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(path, os.pardir))

import logging

class DQN(object):
    def __init__(self, env=None,
                       D=None,
                       model=None,
                       hiddens=[64],
                       learning_rate=1e-3,
                       gamma=0.99,
                       buffer_size=50000,
                       max_timesteps=10**6,
                       print_freq=10,
                       layer_norm=True,
                       exploration_fraction=0.1,
                       exploration_initial_eps=1.0,
                       exploration_final_eps=0.1,
                       target_network_update_freq=500,
                       policy_evaluate_freq=5000,
                       param_noise=True,
                       grad_norm_clipping=10,
                       buffer_batch_size=32,
                       prioritized_replay=False,
                       prioritized_replay_alpha=0.6,
                       prioritized_replay_beta0=0.4,
                       prioritized_replay_beta_iters=None,
                       prioritized_replay_eps=1e-6,
                       action_list=[]):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        D : TODO, optional
        prioritized_replay: True
            if True prioritized replay buffer will be used.
        prioritized_replay_alpha: float
            alpha parameter for prioritized replay buffer
        prioritized_replay_beta0: float
            initial value of beta for prioritized replay buffer
        prioritized_replay_beta_iters: int
            number of iterations over which beta will be annealed from initial value
            to 1.0. If set to None equals to max_timesteps.
        prioritized_replay_eps: float

        """
        if model is None:
            self._model = deepq.models.mlp(hiddens, layer_norm=layer_norm)
        else:
            self._model = model

        if env is not None:
            self._obs_shape = self._env.observation_space.shape
            self._n_action = self._env.action_space.n
        else:
            self._obs_shape = D[0, 0].shape # state
            self._n_action = len(action_list)

        self._env = env
        self._D = D

        self._hiddens = hiddens
        self._lr = learning_rate
        self._gamma = gamma
        self._buffer_size = buffer_size
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
        self._policy_evaluate_freq = policy_evaluate_freq

        self._prioritized_replay = prioritized_replay
        self._prioritized_replay_alpha = prioritized_replay_alpha
        self._prioritized_replay_beta0 = prioritized_replay_beta0
        self._prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self._prioritized_replay_eps = prioritized_replay_eps

        logging.info("dumping D of size {} into experience replay".format(D.shape))
        n_sample = D.shape[0]
        if prioritized_replay:
            self._replay_buffer = PrioritizedReplayBuffer(n_sample, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = max_timesteps
            self._beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            self._replay_buffer = ReplayBuffer(n_sample)

        # now add
        for episode in D:
            s, a, r, s_next, _, done = episode
            self._replay_buffer.add(s, a, r, s_next, float(done))



    def train(self, use_batch=False, reward_fn=None):
        with tf.Graph().as_default():
            if use_batch:
                return self._train_batch(reward_fn=reward_fn)
            else:
                return self._train_online()


    def _train_online(self):
        # Enabling layer_norm here is import for parameter space noise!
        # consider changing this
        act = deepq.learn(
                        self._env,
                        q_func=self._model,
                        lr=self._lr,
                        gamma=self._gamma,
                        max_timesteps=self._max_timesteps,
                        buffer_size=self._buffer_size,
                        exploration_fraction=self._exploration_fraction,
                        exploration_final_eps=self._exploration_final_eps,
                        print_freq=self._print_freq,
                        param_noise=self._param_noise,
                        target_network_update_freq=self._target_network_update_freq,
                        prioritized_replay = self._prioritized_replay,
                        prioritized_replay_alpha = self._prioritized_replay_alpha,
                        prioritized_replay_beta0 = self._prioritized_replay_beta0,
                        prioritized_replay_beta_iters = self._prioritized_replay_beta_iters,
                        prioritized_replay_eps = self._rioritized_replay_eps
                    )
        #print("Saving model to mountaincar_model.pkl")
        act.save("{}/data/mountaincar_model.pkl".format(root_path))
        self._policy = act
        return act



    def _train_batch(self, reward_fn=None):
        sess = tf.Session()
        sess.__enter__()

        def make_obs_ph(name):
            return BatchInput(self._obs_shape, name=name)

        tools = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=self._model,
            num_actions=self._n_action,
            optimizer=tf.train.AdamOptimizer(learning_rate=self._lr),
            gamma=self._gamma,
            grad_norm_clipping=self._grad_norm_clipping
        )
        act, train, update_target, debug = tools


        self._timestep = int(self._exploration_fraction * self._max_timesteps),

        U.initialize()
        update_target()


        for t in itertools.count():
            if self._prioritized_replay:
                experience = self._replay_buffer.sample(self._buffer_batch_size,
                        beta=self._beta_schedule.value(t + 1))
                (s, a, r, s_next, dones, weights, batch_idxes) = experience
                if reward_fn is not None:
                    r = np.array([np.asscalar(reward_fn(s,a)) for s, a in zip(s, a)])
            else:
                s, a, r, s_next, dones = self._replay_buffer.sample(self._buffer_batch_size)
                if reward_fn is not None:
                    r = np.array([np.asscalar(reward_fn(s,a)) for s, a in zip(s, a)])
                weights, batch_idxes = np.ones_like(r), None
            td_errors = train(s, a, r, s_next, dones, weights)

            if self._prioritized_replay:
                new_priorities = np.abs(td_errors) + self._prioritized_replay_eps
                self._replay_buffer.update_priorities(batch_idxes, new_priorities)


            if t % self._target_network_update_freq == 0:
                logging.info("been trained {} steps".format(t))
                update_target()
            if t > 100 and t % self._policy_evaluate_freq == 0:
                logging.info("evaluating the policy...{} steps".format(t))
                if self._env is not None:
                    self._evaluate_policy(act)

            if t > self._max_timesteps:
                break


        self._policy = act
        return act


    def _evaluate_policy(self, act, max_iter=3000):
        episode_rewards = [0.0]
        obs = self._env.reset()
        for t in itertools.count():
            if t > max_iter:
                break
            action = act(obs[None])[0]
            new_obs, rew, done, _ = self._env.step(action)
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = self._env.reset()
                episode_rewards.append(0)


        logger.record_tabular("steps", t)
        logger.record_tabular("episodes", len(episode_rewards))
        logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
        logger.record_tabular("% time spent exploring", int(100 * 0.0))
        logger.dump_tabular()


class DQNSepsis(DQN):
    def __init__(self, D):
        s = D["s"]
        a = D["a"]
        #r = D["r"]
        s_next = D["s_next"]
        absorb = D["done"]
        n_sample = s.shape[0]

        D_mat = []
        for i in range(n_sample):
            t = [s[i], a[i][0], None, s_next[i], None, absorb[i][0]]
            D_mat.append(t)
        D_mat = np.array(D_mat)

        super().__init__(
                         env=None,
                         D=D_mat,
                         hiddens=[128, 64],
                         learning_rate=1e-4,
                         gamma=0.99,
                         buffer_size=n_sample,
                         max_timesteps=5*10**4,
                         print_freq=5000,
                         layer_norm=True,
                         exploration_fraction=0.001,
                         exploration_final_eps=0.001,
                         policy_evaluate_freq=1000,
                         param_noise=True,
                         grad_norm_clipping=10,
                         buffer_batch_size=256,
                         action_list=range(25))


    def solve(self, reward_fn=None, return_policy=True):
        act = self.train(use_batch=True, reward_fn=reward_fn)
        return ActWrapper(act=act)


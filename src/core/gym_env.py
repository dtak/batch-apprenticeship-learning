import sys
import logging
from collections import namedtuple
from itertools import count

import gym
import numpy as np

from src.core import plotting
from src.core.policy import RandomPolicy2

#from src.contrib.baselines.gail import mlp_policy
#from src.contrib.baselines.common import set_global_seeds, tf_util as U

T = namedtuple("Transition", ["s", "a", "r", "s_next", "absorb", "done"])


import types
def new_timelimit_reset(self, s_init=None):
    import time
    self._episode_started_at = time.time()
    self._elapsed_steps = 0
    return self.env.reset(s_init)


def new_reset(self, s_init=None):
    if s_init is None:
        self.state = self.__class__().reset()
    else:
        self.state = np.array(s_init)
    return self.state


class GymSimulator(object):
    """Docstring for OpenAI simulator """

    def __init__(self, env_id, max_iter):
        """TODO: to be defined1.

        Parameters
        ----------
        env_id : str
        """
        env = gym.make(env_id)
        self._env = env
        self._max_iter = max_iter


    def simulate(self, pi, n_episode, reward_fn=None):
        """TODO: Docstring for simulate

        Parameters
        ----------
        pi : Policy
            behavior policy
        n_episode : int

        Returns
        -------
        D: a collection of transition samples

        """

        D = []
        r_list = [0]

        env = self._env

        for epi_i in range(n_episode):

            sys.stdout.flush()
            traj = []
            s = env.reset()

            r_list.append(0)
            for t in count():
                a = pi.choose_action(s)
                s_next, r, done, _ = env.step(a)

                if reward_fn is not None:
                    r = reward_fn(s)

                r_list[-1] += r


                absorb = done and (t + 1) < self._max_iter

                logging.debug("s {} a {} s_next {} r {} absorb {}".format(s, a, r, s_next,
                    absorb))

                transition = T(s=s, a=a, r=r, s_next=s_next, absorb=absorb, done=done)
                traj.append(transition)

                if done:
                    logging.debug("done after {} steps".format(t))
                    break

                s = s_next

                print("\rStep {} @ Episode {}/{} ({})".format(t, epi_i + 1,
                    n_episode, r_list[-2]), end="")

            D.append(traj)

        return D


    def simulate_one_step(self, n_sample, pi=None):
        """TODO: Docstring for simulate_one_step.

        used for collecting diverse transitions

        Parameters
        ----------
        n_sample : int
        pi : Policy

        Returns
        -------
        list

        """

        D = []
        env = self._env

        if pi is None:
            pi = RandomPolicy2(range(env.action_space.n))

        states = np.array([env.observation_space.sample() for _ in range(n_sample)])
        for s in states:
            env.reset(s)
            a = pi.choose_action(s)
            s_next, r, done, _ = env.step(a)
            D.append([[s, a, r, s_next, done]])
        return D


    def simulate_mixed(self, env, pi_list, sample_size, mix_ratio):
        """TODO: Docstring for simulate_mixed.

        Parameters
        ----------
        env : TODO
        pi_list : TODO
        sample_size : TODO
        mix_ratio : TODO

        Returns
        -------
        TODO

        """
        traj_list = []
        for pi, r in zip(pi_list, mix_ratio):
            trajs = self.simulate(pi, n_episode=int(r * sample_size))
            traj_list += trajs
        return traj_list


    @staticmethod
    def to_matrix(D):
        """TODO: Docstring for process_.

        Parameters
        ----------
        o_ma : TODO

        Returns
        -------
        TODO

        """
        D_ = None
        for traj in D:
            if D_ is None:
                D_ = np.array(traj)
            else:
                D_ = np.vstack((D_, np.array(traj)))
        return D_


    @classmethod
    def to_dict(cls, D):
        """convert data to dictionary

        with useful key-values

        Parameters
        ----------
        D : in the format as returned by simulate

        Returns
        -------
        TODO

        """
        D = cls.to_matrix(D)
        data = {}
        data["s"] = np.array(list(D[:, 0]))
        data["a"] = np.array(list(D[:, 1]))
        data["r"] = np.array(list(D[:, 2]))
        data["s_next"] = np.array(list(D[:, 3]))
        data["done"] = np.array(list(D[:, 4]))

        if D.shape[1] == 6:
            data["absorb"] = np.array(list(D[:, 5]))

        return data


def traj_generator(pi, env, reward_fn, horizon, stochastic=False):
    """
    inspired by openai trpo_mpi.py


    a trajectory has the same length because we consider them i.i.d samples
    this may be fine for behavior cloning
    not so fine for irl because we need to compute feature expectation


    Data structure of the input .npz:
    the data is save in python dictionary format with keys:
        'acs', 'ep_rets', 'rews', 'obs'
    the values of each item is a list storing the expert trajectory sequentially
    a transition can be:
        (data['obs'][t], data['acs'][t], data['obs'][t+1])
    and get reward data['rews'][t]
    return a generator that yields a single trajectory each time
    """

    #U.make_session(num_cpu=1).__enter__()
    #set_global_seeds(args.seed)
    ## Setup network
    ## ----------------------------------------
    #ob_space = env.observation_space
    #ac_space = env.action_space
    #def policy_fn(name, ob_space, ac_space, reuse=False):
    #    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    #                                reuse=reuse, hid_size=policy_hidden_size, num_hid_layers=2)
    #pi = policy_func("pi", ob_space, ac_space, reuse=False)
    #U.initialize()
    ## Prepare for rollouts
    ## ----------------------------------------
    #U.load_state(e_pi_path)

    data = {}

    t = 0
    ac = env.action_space.sample()
    new = 1
    rew = 0.0
    true_rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []
    ep_lens = []

    obs = np.array([ob for _ in range(horizon)])
    obs_next = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    vpred = 0 # temp
    news = np.zeros(horizon, 'int32')
    #acs = np.array([ac for _ in range(horizon)])
    # float needed for cont-action task compatibility
    acs = np.array([ac for _ in range(horizon)], 'float32')
    prevacs = acs.copy()

    while True:
        prevac = ac
        #ac, vpred = pi.act(stochastic, ob)
        ac = pi.choose_action(ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "ob_next": obs_next, "rew": rews,
                    "true_rew": true_rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens,
                   "ep_true_rets": ep_true_rets}
            #_, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        rew = reward_fn(ob, ac)
        ob, true_rew, new, _ = env.step(ac)
        rews[i] = rew
        true_rews[i] = true_rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        # to account for episode termination (by reaching goal)
        obs_next[i] = ob
        t += 1






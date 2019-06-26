'''
credit: https://github.com/openai/baselines/blob/master/baselines/gail/dataset/mujoco_dset.py

Data structure of the input .npz:
the data is save in python dictionary format with keys:
    'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be:
    (data['obs'][t], data['acs'][t], data['obs'][t+1])
    and get reward data['rews'][t]
'''

from contrib.baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, obs, acs, obs_next, randomize):
        self.obs = obs
        self.acs = acs
        self.obs_next = obs_next
        assert len(self.obs) == len(self.obs_next)
        assert len(self.obs) == len(self.acs)
        self.randomize = randomize
        self.num_pairs = len(obs)
        self.init_pointer()

    def init_pointer(self):
        """
        shuffle with indices
        """
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.obs = self.obs[idx, :]
            self.acs = self.acs[idx, :]
            self.obs_next = self.obs_next[idx, :]

    def get_next_batch(self, batch_size):
        info = {}
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.obs, self.acs.squeeze(), self.obs_next, info
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        obs = self.obs[self.pointer:end, :]
        acs = self.acs[self.pointer:end, :]
        obs_next = self.obs_next[self.pointer:end, :]
        self.pointer = end

        return obs, acs.squeeze(), obs_next, info


class GymDataset(object):
    def __init__(self, expert_path,
                       train_fraction=0.7,
                       traj_limitation=-1,
                       randomize=True):

        traj_data = np.load(expert_path)

        if traj_limitation < 0:
            traj_limitation = len(traj_data['ob_list'])
        obs = traj_data['ob_list'][:traj_limitation]
        acs = traj_data['ac_list'][:traj_limitation]
        obs_next = traj_data['ob_next_list'][:traj_limitation]


        if len(acs.shape) == 2:
            acs = np.expand_dims(acs, axis=2)


        def flatten(x):
            _, size = x[0].shape
            episode_length = [len(i) for i in x]
            y = np.zeros((sum(episode_length), size))
            start_idx = 0
            for l, x_i in zip(episode_length, x):
                y[start_idx:(start_idx+l)] = x_i
                start_idx += l
                return y
        self.obs = np.array(flatten(obs))
        self.acs = np.array(flatten(acs))
        self.obs_next = np.array(flatten(obs_next))
        self.rets = traj_data['ep_true_ret_list'][:traj_limitation]
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)

        assert len(self.obs) == len(self.acs)
        assert len(self.obs_next) == len(self.acs)

        if len(self.acs.shape) == 1:
            self.acs = np.expand_dims(self.acs, axis=1)
        assert len(self.acs.shape) == 2

        self.num_traj = min(traj_limitation, len(traj_data['ob_list']))
        self.num_transition = len(self.obs)
        self.randomize = randomize

        self.dset = Dset(self.obs, self.acs, self.obs_next, self.randomize)

        idx = int(self.num_transition*train_fraction)
        self.train_set = Dset(self.obs[:idx, :],
                              self.acs[:idx, :],
                              self.obs_next[:idx, :],
                              self.randomize)
        self.val_set = Dset(self.obs[idx:, :],
                            self.acs[idx:, :],
                            self.obs_next[idx:, :],
                            self.randomize)

        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True):
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]

        def flatten(x):
            # x.shape = (E,), or (E, L, D)
            _, size = x[0].shape
            episode_length = [len(i) for i in x]
            y = np.zeros((sum(episode_length), size))
            start_idx = 0
            for l, x_i in zip(episode_length, x):
                y[start_idx:(start_idx+l)] = x_i
                start_idx += l
                return y
        self.obs = np.array(flatten(obs))
        self.acs = np.array(flatten(acs))
        self.rets = traj_data['ep_rets'][:traj_limitation]
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.randomize)
        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


class SepsisDataset(object):
    def __init__(self, expert_path,
                       train_fraction=0.7,
                       traj_limitation=-1,
                       randomize=True):

        traj_data = np.load(expert_path)

        if traj_limitation < 0:
            traj_limitation = len(traj_data['ob_list'])
        obs = traj_data['ob_list']
        acs = traj_data['ac_list']
        obs_next = traj_data['ob_next_list']



        if len(acs.shape) == 2:
            # action is only one dof
            acs = np.expand_dims(acs, axis=2)


        def flatten(x):
            # x.shape = (E,), or (E, L, D)
            _, size = x[0].shape
            episode_length = [len(i) for i in x]
            y = np.zeros((sum(episode_length), size))
            start_idx = 0
            for l, x_i in zip(episode_length, x):
                y[start_idx:(start_idx+l)] = x_i
                start_idx += l
                return y

        self.obs = np.array(obs)
        self.acs = np.array(acs)
        self.obs_next = np.array(obs_next)
        self.rets = traj_data['ep_true_ret_list']
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)

        assert len(self.obs) == len(self.acs)
        assert len(self.obs_next) == len(self.acs)

        if len(self.acs.shape) == 1:
            self.acs = np.expand_dims(self.acs, axis=1)
        assert len(self.acs.shape) == 2

        self.num_traj = min(traj_limitation, len(traj_data['ob_list']))
        self.num_transition = len(self.obs)
        self.randomize = randomize

        self.dset = Dset(self.obs, self.acs, self.obs_next, self.randomize)

        # for behavior cloning
        idx = int(self.num_transition*train_fraction)
        self.train_set = Dset(self.obs[:idx, :],
                              self.acs[:idx, :],
                              self.obs_next[:idx, :],
                              self.randomize)
        self.val_set = Dset(self.obs[idx:, :],
                            self.acs[idx:, :],
                            self.obs_next[idx:, :],
                            self.randomize)

        self.log_info()


    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)



def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)


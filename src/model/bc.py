'''
original implementation credit: https://github.com/openai/baselines

heavily adapted to suit our needs.
'''

import argparse
import tempfile
import os.path as osp
import gym
import logging
from tqdm import tqdm

import tensorflow as tf
import numpy as np

import os
import sys
import glob

file_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.abspath(os.path.join(file_path, os.pardir))
root_path = os.path.abspath(os.path.join(src_path, os.pardir))
sys.path.insert(0, src_path)

from contrib.baselines.gail import mlp_policy
from contrib.baselines import bench
from contrib.baselines import logger
from contrib.baselines.common import set_global_seeds, tf_util as U
from contrib.baselines.common.misc_util import boolean_flag
from contrib.baselines.common.mpi_adam import MpiAdam
from core.data_util import GymDataset, SepsisDataset
from core.run_gym import run_gym


def learn_original(pi, dataset, env_name, n_action, prefix, traj_lim, seed,
          optim_batch_size=128, max_iters=5e3,
          adam_epsilon=1e-4, optim_stepsize=1e-4,
          ckpt_dir=None, plot_dir=None, task_name=None,
          verbose=False):
    """
    learn without regularization
    """
    # custom hyperparams
    seed = 0
    max_iters = 5e4


    val_per_iter = int(max_iters/10)
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(tf.to_float(ac-pi.ac)))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()
    logger.log("Training a policy with Behavior Cloning")
    logger.log("with {} trajs, {} steps".format(dataset.num_traj, dataset.num_transition))


    loss_history = {}
    loss_history["train_action_loss"] = []
    loss_history["val_action_loss"] = []


    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert, _, _ = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert, _, _ = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))

            loss_history["train_action_loss"].append(train_loss)
            loss_history["val_action_loss"].append(val_loss)

    plot(env_name, loss_history, traj_lim, plot_dir)

    os.makedirs(ckpt_dir, exist_ok=True)
    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        ckpt_fname = "ckpt.bc.{}.{}".format(traj_lim, seed)
        savedir_fname = osp.join(ckpt_dir, ckpt_fname)
    U.save_state(savedir_fname, var_list=pi.get_variables())
    return savedir_fname


def learn(network, dataset, env_name, n_action, prefix, traj_lim, seed,
          optim_batch_size=32, max_iters=1e4,
          adam_epsilon=1e-4, optim_stepsize=3e-4,
          ckpt_dir=None, plot_dir=None, task_name=None,
          verbose=False):
    """
    learn with regularization
    """
    seed = 0
    alpha = 0.7
    beta = 1.0

    pi = network.pi
    T = network.T

    val_per_iter = int(max_iters/20)

    ob = U.get_placeholder_cached(name="ob")
    T_ac = U.get_placeholder_cached(name="T_ac")
    pi_stochastic = U.get_placeholder_cached(name="pi_stochastic")
    T_stochastic = U.get_placeholder_cached(name="T_stochastic")

    ac = network.pdtype.sample_placeholder([None])
    ob_next = network.ob_next_pdtype.sample_placeholder([None])

    onehot_ac = tf.one_hot(ac, depth=n_action)
    ce_loss = tf.losses.softmax_cross_entropy(logits=pi.logits,
            onehot_labels=onehot_ac)

    ce_loss = tf.reduce_mean(ce_loss)

    reg_loss = tf.reduce_mean(tf.square(tf.to_float(ob_next-network.ob_next)))

    losses = [ce_loss, reg_loss]

    total_loss = alpha * ce_loss + beta * reg_loss

    var_list = network.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, T_ac, ob_next, pi_stochastic, T_stochastic],
            losses +[U.flatgrad(total_loss, var_list)])

    U.initialize()
    adam.sync()
    logger.log("Training a policy with Behavior Cloning")
    logger.log("with {} trajs, {} steps".format(dataset.num_traj, dataset.num_transition))


    loss_history = {}
    loss_history["train_action_loss"] = []
    loss_history["train_transition_loss"] = []
    loss_history["val_action_loss"] = []
    loss_history["val_transition_loss"] = []

    for iter_so_far in tqdm(range(int(max_iters))):
        #ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        ob_expert, ac_expert, ob_next_expert, info = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss_ce, train_loss_reg, g = lossandgrad(ob_expert, ac_expert, ac_expert, ob_next_expert, True, True)
        adam.update(g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            #ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            ob_expert, ac_expert, ob_next_expert, info = dataset.get_next_batch(-1, 'val')

            val_loss_ce, val_loss_reg,  _ = lossandgrad(ob_expert, ac_expert, ac_expert, ob_next_expert, True, True)
            items = [train_loss_ce, train_loss_reg, val_loss_ce, val_loss_reg]
            logger.log("Training Action loss: {}\n" \
                       "Training Transition loss: {}\n" \
                       "Validation Action loss: {}\n" \
                       "Validation Transition Loss:{}\n".format(*items))
            loss_history["train_action_loss"].append(train_loss_ce)
            loss_history["train_transition_loss"].append(train_loss_reg)
            loss_history["val_action_loss"].append(val_loss_ce)
            loss_history["val_transition_loss"].append(val_loss_reg)

            #if len(loss_history["val_action_loss"]) > 1:
            #    val_loss_ce_delta = loss_history["val_action_loss"][-1] - val_loss_ce
            #    if np.abs(val_loss_ce_delta) < val_stop_threshold:
            #        logger.log("validation error seems to have converged.")
            #        break


    plot(env_name, loss_history, traj_lim, plot_dir)

    os.makedirs(ckpt_dir, exist_ok=True)
    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        ckpt_fname = "ckpt.bc.{}.{}".format(traj_lim, seed)
        savedir_fname = osp.join(ckpt_dir, ckpt_fname)
    U.save_state(savedir_fname, var_list=network.get_variables())
    return savedir_fname


def plot(env_name, loss, traj_lim, save_path):
    """TODO: Docstring for plot.

    Parameters
    ----------
    arg1 : TODO

    Returns
    -------
    TODO

    """
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    num_data= len(loss["train_action_loss"])
    #plt.ylim([0, 0.1])
    plt.ylabel('loss')
    plt.title('pretraining loss for {}'.format(env_name))
    plt.plot(np.arange(num_data), loss["train_action_loss"], c="r",
        linestyle="--")
    plt.plot(np.arange(num_data), loss["val_action_loss"], c="r")
    if "train_transition_loss" in loss:
        plt.plot(np.arange(num_data), loss["train_transition_loss"], c="b", linestyle="--")
        plt.plot(np.arange(num_data), loss["val_transition_loss"], c="b")
        plt.legend(['train_action', 'train_transition', 'val_action', 'val_transition'], loc='best')
    plt.legend(['train_action', 'val_action'], loc='best')
    plt.savefig(os.path.join(save_path, "loss.{}.{}.png".format(env_name,
        traj_lim)), format="png")
    plt.close()


def train_bc(task, params, ob_space, ac_space, args, env):
    task_path = os.path.join(root_path, "task", args.task)
    plot_path = os.path.join(task_path, "result")

    dataset = GymDataset(expert_path=args.expert_path,
                      traj_limitation=args.traj_limitation)


    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name,
                                    ob_space=ob_space,
                                    ac_space=ac_space,
                                    reuse=reuse,
                                    hid_size_phi=args.policy_hidden_size,
                                    num_hid_layers_phi=2,
                                    dim_phi=args.dim_phi)
    env_name = task["env_id"]
    name = "pi.{}.{}".format(env_name.lower().split("-")[0], args.traj_limitation)
    pi = policy_fn(name, ob_space, ac_space)
    n_action = env.action_space.n



    fname = "ckpt.bc.{}.{}".format(args.traj_limitation, args.seed)
    savedir_fname  = osp.join(args.checkpoint_dir, fname, fname)

    if not os.path.exists(savedir_fname + ".index"):
        savedir_fname = learn(pi,
                              dataset,
                              env_name,
                              n_action,
                              prefix="bc",
                              seed=args.seed,
                              traj_lim=args.traj_limitation,
                              max_iters=args.BC_max_iter,
                              ckpt_dir=osp.join(args.checkpoint_dir, fname),
                              plot_dir=plot_path,
                              task_name=task["env_id"],
                              verbose=True)
        logger.log(savedir_fname + "saved")



#    avg_len, avg_ret = run_gym(env,
#                               policy_fn,
#                               savedir_fname,
#                               timesteps_per_batch=args.horizon,
#                               number_trajs=10,
#                               stochastic_policy=args.stochastic_policy,
#                               save=args.save_sample,
#                               reuse=True)
#
#
    return savedir_fname


def train_bc_sepsis(task, params, ob_space, ac_space, args):
    task_path = os.path.join(root_path, "task", args.task)
    plot_path = os.path.join(task_path, "result")

    dataset = SepsisDataset(expert_path=args.expert_path,
                            traj_limitation=args.traj_limitation)


    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)

    # just im
    #def policy_fn(name, ob_space, ac_space, reuse=False):
    #    return mlp_policy.MlpPolicyOriginal(name=name,
    #                                ob_space=ob_space,
    #                                ac_space=ac_space,
    #                                reuse=reuse,
    #                                hid_size=args.policy_hidden_size,
    #                                num_hid_layers=2)


    # im + reg

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name,
                                    ob_space=ob_space,
                                    ac_space=ac_space,
                                    reuse=reuse,
                                    hid_size_phi=args.policy_hidden_size,
                                    num_hid_layers_phi=2,
                                    dim_phi=args.dim_phi)




    env_name = task["env_id"]
    name = "pi.{}.{}".format(env_name.lower().split("-")[0], args.traj_limitation)
    pi = policy_fn(name, ob_space, ac_space)
    n_action = ac_space.n


    fname = "ckpt.bc.{}.{}".format(args.traj_limitation, args.seed)
    savedir_fname  = osp.join(args.checkpoint_dir, fname, fname)

    if not os.path.exists(savedir_fname + ".index"):
        #savedir_fname = learn_original(pi,
        #                      dataset,
        #                      env_name,
        #                      n_action,
        #                      prefix="bc",
        #                      seed=args.seed,
        #                      traj_lim=args.traj_limitation,
        #                      max_iters=args.BC_max_iter,
        #                      ckpt_dir=osp.join(args.checkpoint_dir, fname),
        #                      plot_dir=plot_path,
        #                      task_name=task["env_id"],
        #                      verbose=True)


        savedir_fname = learn(pi,
                              dataset,
                              env_name,
                              n_action,
                              prefix="bc",
                              seed=args.seed,
                              traj_lim=args.traj_limitation,
                              max_iters=args.BC_max_iter,
                              ckpt_dir=osp.join(args.checkpoint_dir, fname),
                              plot_dir=plot_path,
                              task_name=task["env_id"],
                              verbose=True)
        logger.log(savedir_fname + "saved")



#    avg_len, avg_ret = run_gym(env,
#                               policy_fn,
#                               savedir_fname,
#                               timesteps_per_batch=args.horizon,
#                               number_trajs=10,
#                               stochastic_policy=args.stochastic_policy,
#                               save=args.save_sample,
#                               reuse=True)

    return savedir_fname


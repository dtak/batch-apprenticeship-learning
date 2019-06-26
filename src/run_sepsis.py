"""
evaluation module for sepsis-related tasks.
this was coded up fast so expect bugs
"""
import os
import sys
import argparse
import pickle
import json

import gym
import numpy as np

from core.evaluator import ALEvaluator
from core.basis import GaussianKernel, RBFKernel2
from core.feature_expectation import MCMuEstimator, LSTDMuEstimator, DeepMuEstimator
from core.policy import RandomPolicy2
from core.data_util import GymDataset

from contrib.scirl import train_scirl
from contrib.baselines import bench, logger
from contrib.baselines.common import set_global_seeds
from contrib.baselines.common.misc_util import boolean_flag
from contrib.baselines.gail import mlp_policy
from contrib.baselines.common import set_global_seeds, tf_util as U

from model.mma import train_mma
from model.bc import train_bc_sepsis
from model.lstd import LSTDMu, LSPI
from model.dqn import DQNSepsis


file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.insert(0, root_path)

import logging
from logging import config

with open(os.path.join(root_path, "context.json"), "r") as f:
    context = json.load(f)

ENV_LIST = context["env"]
MODEL_LIST = context["model"]


def argsparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("task", type=str, choices=ENV_LIST)
    parser.add_argument("--model_id", type=str, choices=MODEL_LIST)
    parser.add_argument("--n_e", type=int, help="demo expert sample size")
    parser.add_argument("--n_ne", type=int, help="demo no expert sample size")
    parser.add_argument("--save_path", type=str, help="save to txt file")
    parser.add_argument("--seed", type=int, default=0, help="seed to keep experiment results in sync")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite best param if available")

    parser.add_argument('--env_id', help='environment ID', default='MountainCar-v0')
    parser.add_argument("--e_filename", type=str, default="demo.stochastic.expert.train.npz")
    parser.add_argument('--task_mode', type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='task')
    parser.add_argument('--horizon', help='Finite Horizon for a traj', type=int,
            default=200)
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--traj_limitation', type=int, default=1)
    parser.add_argument('--policy_hidden_size', type=int, default=64)
    parser.add_argument('--dim_phi', type=int, default=64)
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC',
            type=int, default=1e5)

    boolean_flag(parser, 'stochastic_policy', default=False,
            help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False,
            help='save the trajectories or not')
    boolean_flag(parser, 'pretrain', default=True,
            help='pretrain a poilcy and phi with imitation learning')

    return parser.parse_args()


def main(args):
    set_global_seeds(args.seed)

    task_path = os.path.join(root_path, "task", args.task)
    data_path = os.path.join(task_path, "data")

    task_desc_path = os.path.join(task_path, "task.json")

    with open(task_desc_path, "r") as f:
        task_desc = json.load(f)

    e_path_async = os.path.join(data_path, args.e_filename)
    D_e_bc = np.load(e_path_async)


    data = {}
    data["irl"] = D_e_bc

    model_path = os.path.join(task_path, "model", "{}.json".format(args.model_id))
    with open(model_path, "r") as f:
        params = json.load(f)

    if args.task_mode == "train":
        train(data, task_desc, params, args, task_path)

    elif args.task_mode == "evaluate":
        evaluate(data, task_desc, params, args, task_path)



def train(data, task_desc, params, args, task_path):
    import gym
    ob_dim = data["irl"]["ob_list"][0].shape[0]
    #ob_dim = ob_dim // 4
    c = np.max([np.abs(np.min(data["irl"]["ob_list"])),
                np.abs(np.max(data["irl"]["ob_list"]))])
    ob_low = np.ones(ob_dim) * -c
    ob_high = np.ones(ob_dim) * c
    ob_space = gym.spaces.Box(low=ob_low, high=ob_high)
    n_action = 5
    ac_space = gym.spaces.Discrete(n=n_action)

    if args.pretrain:
        model_path = os.path.join(root_path, "task", args.task, "model")
        fname = "ckpt.bc.{}.{}".format(args.traj_limitation, args.seed)
        ckpt_dir = os.path.join(model_path, fname)
        pretrained_path = os.path.join(ckpt_dir, fname)

        if not os.path.exists(os.path.join(ckpt_dir, "checkpoint")):
            print("==== pretraining starts ===")
            pretrained_path = train_bc_sepsis(task_desc,
                                     params,
                                     ob_space,
                                     ac_space,
                                     args)


        U.make_session(num_cpu=1).__enter__()
        set_global_seeds(args.seed)


        def mlp_pi_wrapper(name, ob_space, ac_space, reuse=False):
            return mlp_policy.MlpPolicy(name=name,
                                        ob_space=ob_space,
                                        ac_space=ac_space,
                                        reuse=reuse,
                                        hid_size_phi=args.policy_hidden_size,
                                        num_hid_layers_phi=2,
                                        dim_phi=args.dim_phi)

        # just imitation learning
        #def mlp_pi_wrapper(name, ob_space, ac_space, reuse=False):
        #    return mlp_policy.MlpPolicyOriginal(name=name,
        #                                    ob_space=ob_space,
        #                                    ac_space=ac_space,
        #                                    reuse=reuse,
        #                                    hid_size=args.policy_hidden_size,
        #                                    num_hid_layers=2)


        env_name = task_desc["env_id"]
        scope_name = "pi.{}.{}".format(env_name.lower().split("-")[0], args.traj_limitation)

        pi_bc = mlp_pi_wrapper(scope_name, ob_space, ac_space)
        U.initialize()
        U.load_state(pretrained_path)
        phi_bc = pi_bc.featurize

        def phi_old(s, a):
            """
            TODO: if action is discrete
            one hot encode action and concatenate with phi(s)
            """
            # expect phi(s) -> (N, state_dim)
            # expect a -> (N, action_dim)
            phi_s = phi_bc(s)


            if len(phi_s.shape) == 1:
                # s -> (1, state_dim)
                phi_s = np.expand_dims(phi_s, axis=0)

            # if a = 5
            try:
                if a == int(a):
                    a = [a]
            except:
                pass

            a = np.array(a)
            # if a = [5]
            if len(a.shape) == 1:
                a = np.expand_dims(a, axis=1)
            # otherwise if a = [[5], [3]]
            phi_sa = np.hstack((phi_s, a))
            return phi_sa


        def phi_discrete_action(n_action):
            def f(s, a):
                # expect phi(s) -> (N, state_dim)
                # expect a -> (N, action_dim)
                phi_s = phi_bc(s)

                try:
                    if a == int(a):
                        a = [a]
                except:
                    pass
                a = np.array(a)

                a_onehot = np.eye(n_action)[a.astype(int)]


                if len(phi_s.shape) == 1:
                    # s -> (1, state_dim)
                    phi_s = np.expand_dims(phi_s, axis=0)

                try:
                    phi_sa = np.hstack((phi_s, a_onehot))
                except:
                    a_onehot = a_onehot.reshape(a_onehot.shape[0],
                            a_onehot.shape[2])
                    phi_sa = np.hstack((phi_s, a_onehot))
                return phi_sa
            return f


        if isinstance(ac_space, gym.spaces.Discrete):
            phi = phi_discrete_action(ac_space.n)
        elif isinstance(ac_space, gym.spaces.Box):
            phi = phi_continuous_action
        else:
            raise NotImplementedError


    D = data["irl"]
    obs = D["ob_list"].reshape(-1, D["ob_list"].shape[-1])
    obs_p1 = D["ob_next_list"].reshape(-1, D["ob_next_list"].shape[-1])
    #assuming action dof of 1
    acs = D["ac_list"].reshape(-1)
    new = D["new"].reshape(-1)

    data = {}
    data["s"] = obs
    data["a"] = acs
    data["s_next"] = obs_p1
    data["done"] = data["absorb"] = new

    data["phi_sa"] = phi(obs, acs)
    data["phi_fn"] = phi
    data["phi_fn_s"] = phi_bc

    data["psi_sa"] = data["phi_sa"]
    data["psi_fn"] = phi

    evaluator = ALEvaluator(data, task_desc["gamma"], env=None)
    data_path = os.path.join(task_path, "data")


    pi_0 = pi_bc

    phi_dim = data["phi_sa"].shape[1]

    model_id = "{}.{}".format(params["id"], params["version"])

    if model_id == "mma.0":
        result = train_mma(pi_0, phi_dim, task_desc, params, data, evaluator, ob_space, ac_space)
    elif model_id == "mma.1":
        result = train_mma(pi_0, phi_dim, task_desc, params, data, evaluator, ob_space, ac_space)
    elif model_id == "mma.2":
        #result = train_scirl_v2(data, phi_bc, evaluator, phi_dim, task_desc, params)
        #result = train_scirl_v3(data, phi_bc, evaluator)
        result = train_scirl(data, phi_bc, evaluator)
    else:
        raise NotImplementedError


    name = "{}.{}.{}".format(model_id, args.n_e, args.seed)
    result_path = os.path.join(args.save_path, name + "train.log")

    with open(result_path, "w") as f:
        #flush?
        for step in range(params["n_iteration"] + 1):
            data_points = [step,
                           round(result["margin_mu"][step],2),
                           round(result["margin_v"][step],2),
                           round(result["a_match"][step],2)
                           ]
            f.write("{}\t{}\t{}\t{}\n".format(*data_points))


    with open(os.path.join(args.save_path, name + ".pkl") , "wb") as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    #setup_logging(default_level=logging.INFO)
    args = argsparser()
    main(args)


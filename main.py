"""
entry script for openai related tasks.
for sepsis, use src/run_sepsis.py

Note there may be a few parts that should be handled manually.
"""
import os
import json
import argparse
from subprocess import Popen
import pickle

import logging
from logging import config
from logger import *


python_cmd = "python3"
root_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(root_path, "src")
run_experiment_path = os.path.join(src_path, "run.py")


with open(os.path.join(root_path, "context.json"), "r") as f:
    context = json.load(f)
ENV_LIST = context["env"]
MODEL_LIST = context["model"]


def run_experiment(seed, args, save_path, e_path, ckpt_path):
    "task parallize over multiple trials"
    for i, limit in enumerate(context['traj_limitation']):
        cmd = [python_cmd, run_experiment_path, args.task,
                "--model_id", args.model_id, "--n_e", str(args.n_e),
                "--n_ne", str(args.n_ne), "--seed", str(0),
                "--save_path", save_path,
                "--expert_path", e_path,
                "--checkpoint_dir", ckpt_path,
                "--traj_limitation", str(limit)
                ]

        #for key, val in params.items():
        #    cmd += ["--{}".format(key), str(val)]

        p = Popen(cmd)
        p.communicate()


def main():
    logging.info("Running experiments with the following setup")

    parser = argparse.ArgumentParser()

    parser.add_argument("task", type=str, choices=ENV_LIST)
    parser.add_argument("--model_id", type=str, choices=MODEL_LIST)
    parser.add_argument("--n_trial", type=int, default=1)
    parser.add_argument("--n_e", type=int, default=100, help="expert demo sample size")
    parser.add_argument("--n_ne", type=int, default=100, help="random demo sample size")
    parser.add_argument("--ne_filename", type=str, default="demo.stochastic.random.npz")
    parser.add_argument("--e_filename", type=str, default="demo.deterministic.expert.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite best param if available")


    args = parser.parse_args()

    task_path = os.path.join(root_path, "task", args.task)
    result_path = os.path.join(task_path, "result")
    os.makedirs(task_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(task_path, "model"), exist_ok=True)


    task_desc_path = os.path.join(task_path, "task.json")

    with open(task_desc_path, "r") as f:
        task_desc = json.load(f)


    data_path = os.path.join(task_path, "data")
    os.makedirs(data_path, exist_ok=True)

    #ne_path = os.path.join(data_path, args.ne_filename)
    #if not os.path.exists(ne_path):
    #    raise Exception("Non-expert Demo data does not exist at {}".format(ne_path))

    expert_pi_path = os.path.join(data_path, "{}.pkl".format(task_desc["env_id"]))
    #expert_fname = "{}.trpo.expert".format(task_desc["env_id"].lower())
    #expert_pi_path = os.path.join(data_path, expert_fname)
    e_path = os.path.join(data_path, args.e_filename)

    if not os.path.exists(e_path):
        raise Exception("Expert demo data does not exist at {}".format(e_path))

    ckpt_path = os.path.join(task_path, "model")
    for i in range(args.n_trial):
        run_experiment(i, args, result_path, e_path, ckpt_path)


if __name__ == "__main__":
    setup_logging(default_level=logging.INFO)
    main()

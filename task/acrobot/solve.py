import sys
import os
import json

file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(file_path, "../..")
sys.path.insert(0, root_path)

import os
import pickle

import gym
import sklearn
from sklearn import preprocessing
import numpy as np

from src.contrib.baselines import deepq
from src.core.evaluator import ALEvaluator
from src.core.actwrapper import ActWrapper


import pprint


file_path = os.path.dirname(os.path.realpath(__file__))
task_desc_path = os.path.join(file_path, "task.json")


def main():
    with open(task_desc_path, "r") as f:
        task = json.load(f)

    print("solving" + task["env_id"] + "to obtain expert policy")
    env = gym.make(task["env_id"])


    pprint.pprint(env.__dict__)
    model_path = os.path.join(file_path, "data", task["env_id"] + ".pkl")
    if os.path.exists(model_path):
        act = deepq.load(model_path)
    else:
        model = deepq.models.mlp([64])
        act = deepq.learn(
            env,
            q_func=model,
            lr=1e-3,
            max_timesteps=150000,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            print_freq=100
        )
        act.save(model_path)

    pi_expert = ActWrapper(act)

    # ======= save expert policy and perf =======
    perf = ALEvaluator.evaluate_perf_env(pi_expert, env, task["gamma"])
    print("expert performance", perf)

    log_path = os.path.join(file_path, "data", "expert.log")

    with open(log_path, 'w') as f:
        f.write("v_expert\n{}\n".format(perf))

if __name__ == "__main__":
    main()

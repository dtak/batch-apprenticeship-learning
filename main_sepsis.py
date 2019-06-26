import pickle
import numpy as np
import os
import pandas as pd

DATA_PATH = "data/D_train.pkl"
DATA_PCA_PATH = "data/D_train_pca.pkl"
DATA_COMPACT_PATH = "data/D_train_compact.pkl"
DATA_SIGMOID_COMPACT_PATH = "data/D_train_sigmoid_compact.pkl"
DATA_SIGMOID_PATH = "data/D_train_sigmoid.pkl"

def stack_observation(ob_list, timestep_list, stack_horizon):
    """
    ob_list : numpy.array
        array of dimension (N, ob_dim)
    """
    stacked_ob_list = []
    N, ob_dim = ob_list.shape
    for i, (ob, step) in enumerate(zip(ob_list, timestep_list)):
        stacked = np.array([0.0] * ob_dim * stack_horizon)
        if step == 1:
            # if ob has no previous timesteps
            stacked[-ob_dim:] = ob
        elif step < stack_horizon:
            # if ob has some < T previous timesteps
            # step == 2
            stacked[-ob_dim * step:] = ob_list[i-step+1:i+1, :].flatten()
        elif step >= stack_horizon:
            # if ob has T previous timesteps
            stacked = ob_list[i-4+1:i+1, :].flatten()
        else:
            raise Exception("unknown case")
        stacked_ob_list.append(stacked)
    stacked_ob_list = np.array(stacked_ob_list)
    assert stacked_ob_list.shape == (N, ob_dim*stack_horizon)
    return stacked_ob_list


def get_next_observation(ob_list, timestep_list):
    N, ob_dim =  ob_list.shape
    ob_next_list = np.zeros((N, ob_dim))

    for i, step in enumerate(timestep_list[:-1]):
        next_step = timestep_list[i+1]
        if next_step == 1:
            # if terminal step
            #ob_next_list[i] = np.zeros(ob_dim, dtype=np.float)
            continue
        else:
            # < terminal steps
            ob_next_list[i] = ob_list[i+1]

    assert ob_next_list.shape == (N, ob_dim)
    return ob_next_list


def get_data():
    root_path = "task/sepsis"
    feature_type = ""
    if feature_type == "pca":
        path = os.path.join(root_path, DATA_PCA_PATH)
    elif feature_type == "compact":
        path = os.path.join(root_path, DATA_COMPACT_PATH)
    elif feature_type == "sigmoid_compact":
        path = os.path.join(root_path, DATA_SIGMOID_COMPACT_PATH)
    elif feature_type == "sigmoid":
        path = os.path.join(root_path, DATA_SIGMOID_PATH)
    else:
        path = os.path.join(root_path, DATA_PATH)

    with open(path, "rb") as f:
        D = pickle.load(f)

    D_s_features = D["features"]

    df = pd.read_csv("task/sepsis/data/cleansed_data_train.csv")
    a_iv = np.expand_dims(df["action_iv"].tolist(), axis=1)

    # add iv fluid as part of observation
    D["s"] = np.hstack([D["s"], a_iv])
    timestep_list = np.array(df["bloc"].tolist()).astype(int)

    s = D["s"].astype(np.float)

    # we don't stack as it's not helpful (surprise!)
    # s = stack_observation(s, timestep_list, 4)

    # compose next_s
    # ignore the original D["s_next"]
    # as it's hard to refigure for our use case
    s_next = get_next_observation(s, timestep_list)

    # include vasopressor as action
    # a = np.vstack(D["a"]).astype(np.int)
    a = df["action_vaso"].tolist()
    r = np.vstack(D["r"]).astype(np.float)
    s_next[s_next == None] = 0.
    s_next[np.isnan(s_next)] = 0.
    s_next = s_next.astype(np.float)
    assert np.sum(s_next == None) == 0
    assert np.sum(np.isnan(s_next)) == 0

    done = np.vstack(D["done"]).astype(np.int)
    phi_s = D["phi_s"]
    phi_s_next = D["phi_s_next"]



    if feature_type in ["sigmoid", "sigmoid_compact"]:
        phi_criteria = None
    phi_criteria = np.array(D["phi_criteria"]).astype(np.float)

    # save data in nice format
    data = {}
    data["ob_list"] = s
    data["ob_next_list"] = s_next
    data["ac_list"] = a
    data["ep_true_ret_list"] = r
    data["true_rew_list"] = r
    data["new"] = done
    data["mu_exp"] = D["mu_exp"]

    return data



if __name__ == "__main__":
    """
    add iv fluids as observation like mechvent
    stack observations over four timesteps
    """

    data = get_data()

    data_path = "task/sepsis/data/demo.stochastic.expert.train.npz"
    np.savez(data_path, **data)


    from subprocess import Popen

    python_cmd = "python3"
    root_path = os.path.dirname(os.path.realpath(__file__))
    src_path = os.path.join(root_path, "src")
    run_experiment_path = os.path.join(src_path, "run_sepsis.py")
    task_path = os.path.join(root_path, "task/sepsis")
    ckpt_path = os.path.join(task_path, "model")
    save_path = os.path.join(task_path, "result")

    model_id = "mma.1"

    cmd = [python_cmd, run_experiment_path, "sepsis",
            "--model_id", model_id, "--n_e", "1000",
            "--n_ne", "1000", "--seed", str(0),
            "--save_path", save_path,
            "--expert_path", data_path,
            "--checkpoint_dir", ckpt_path
            ]

    p = Popen(cmd)
    p.communicate()



import matplotlib
from matplotlib import collections  as mc
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='white', palette='Blues')


import numpy as np
import pandas as pd
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D

from time import time as t

import os
notebook_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(notebook_path, os.pardir))

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])


def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20, show=True):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    #Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
    Z = np.apply_along_axis(lambda _: np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0, alpha=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    #ax.set_title("Mountain \"Cost To Go\" Function")
    ax.set_title("Mountain Value Function")
    fig.colorbar(surf)
    fig.savefig("{}/data/mc_value_fn_{}.png".format(root_path, t()), ppi=300, bbox_inches='tight')

    if show:
        plt.show(fig)
    else:
        plt.close(fig)



def plot_value_function(V, title="Value Function", show=True):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))

    if show:
        plt.show(fig)
    else:
        plt.close(fig)


def plot_episode_stats(stats, smoothing_window=10, show=True):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    #fig1.savefig("{}/data/episode_len_{}.png".format(root_path, t()), ppi=300, bbox_inches='tight')
    if show:
        plt.show(fig1)
    else:
        plt.close(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    #fig2.savefig("{}/data/episode_reward_{}.png".format(root_path, t()), ppi=300, bbox_inches='tight')
    if show:
        plt.show(fig2)
    else:
        plt.close(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    #fig3.savefig("{}/data/episode_t_epi_{}.png".format(root_path, t()), ppi=300, bbox_inches='tight')
    if show:
        plt.show(fig3)
    else:
        plt.close(fig3)

    return fig1, fig2, fig3


def plot_trajectory_mountain_car(D, show=True):
    """TODO: Docstring for plot_trajectories.

    Parameters
    ----------
    arg1 : TODO

    Returns
    -------
    TODO

    """
    fig = plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=15.0)
    plt.yticks(fontsize=15.0)

    ax = plt.gca()
    lines = [[(-0.4, 0.0), (-0.6, 0.0)]]
    c = np.array([(1, 0, 0, 1)])
    lc = mc.LineCollection(lines, colors='red', linewidths=5)
    ax.add_collection(lc)

    for episode in D:
        states = []
        for (s, a, r, s_next, absorb, done) in episode:
            states.append(s)
        states = np.array(states)
        sc = ax.scatter(states[:,0], states[:,1], c=range(len(states[:,0])), cmap=plt.get_cmap("YlOrRd"), s=7.0)
    ax.set_xlim(-1.3, 0.6)
    ax.axvline(0.5, c='blue', linewidth=5)
    ax.set_xlabel('Position', fontsize=20.0)
    ax.set_ylabel('Velocity', fontsize=20.0)

    if show:
        plt.show(fig)
    else:
        plt.close(fig)




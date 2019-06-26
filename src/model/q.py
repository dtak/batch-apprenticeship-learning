import numpy as np
from policy import EpsilonGreedyPolicy, GreedyPolicy

class Q(EpsilonGreedyPolicy):
    """Docstring for QLearning. """

    def __init__(self, n_states, n_actions, epsilon, gamma, alpha):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        epsilon : TODO
        gamma : TODO
        n_trials : TODO
        n_episodes : TODO
        max_iter : TODO


        """
        super().__init__(self, n_states, n_actions, epsilon)
        self._gamma = gamma
        self._alpha = alpha


    def update_Q(self, s, a, r, s_next):
        Q = self.Q
        best_a = self.choose_action(s_next)
        td_target = r + self._gamma * Q[s_next, best_a]
        td_error = td_target - Q[s, a]
        Q_sa_new = Q[s, a] + self._alpha * td_error
        self.update_Q_val(s, a, Q_sa_new)




if __name__ == "__main__":
    """
    example of using QLearning
    """
    import gym
    env = gym.envs.make("MountainCarContinuous-v0")
    n_states = 100
    n_actions = 100
    n_trials = 1
    n_episodes = 100
    max_iter = 1000
    gamma = 0.80
    alpha = 0.01

    Q = Q(n_states, n_actions, epsilon, gamma, alpha)

    """
    for now, suppose env is openai gym
    """
    # Loop over some number of episodes
    state_count = env.num_states
    action_count = env.num_actions
    reward_per_episode = np.zeros((trial_count, episode_min_count))
    reward_per_step = np.zeros((trial_count, global_min_iter_count))
    # episode gets terminated when past local_max

    trial_lengths = []
    for trial_idx in range(trial_count):

        # Initialize the Q table
        Q_table = np.zeros((state_count, action_count))
        transition_count_table = np.zeros((state_count, state_count))
        reward_value_table = np.zeros((state_count))

        global_iter_idx = 0

        # Loop until the episode is done
        for episode_idx in range(episode_min_count):
            # print('episode count {}'. format(episode_idx))
            # print('global iter count {}'. format(global_iter_idx))

            local_iter_idx = 0
            # Start the env
            env.reset()
            state = env.observe()
            action = policy(state, Q_table, action_count, epsilon)
            episode_reward_list = []

            # Loop until done
            while local_iter_idx < local_max_iter_count:
                # print('local iter count {}'. format(local_iter_idx))

                new_state, reward = env.perform_action(action)
                new_action = policy(new_state, Q_table, action_count, epsilon)

                # FILL IN HERE: YOU WILL NEED CASES FOR THE DIFFERENT ALGORITHMS

                if 'alpha' in hyperparams:
                    alpha = hyperparams['alpha'](hyperparams, np.cbrt(global_iter_idx+1))

                Q_table = update_Q_Qlearning(Q_table, alpha, gamma, state, action, reward, new_state)

                # store the data
                episode_reward_list.append(reward)
                if global_iter_idx < global_min_iter_count:
                    reward_per_step[trial_idx, global_iter_idx] = reward


                # stop if at goal/else update for the next iteration
                if env.is_terminal(state):
                    break
                else:
                    state = new_state
                    action = new_action

                local_iter_idx += 1
                global_iter_idx += 1

            # Store the rewards
            reward_per_episode[trial_idx, episode_idx] = np.sum(episode_reward_list)

        trial_lengths.append(global_iter_idx)
        reward_per_step[trial_idx, :] = np.cumsum(reward_per_step[trial_idx, :])

    # slice off to the shortest trial for consistent visualization
    reward_per_step = reward_per_step[:,:np.min(trial_lengths)]
    return Q_table, reward_per_step, reward_per_episode



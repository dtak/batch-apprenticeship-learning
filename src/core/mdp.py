class MDP(object):
    """Docstring for MDP. """

    def __init__(self, env):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO

        """
        self._env = env
        self._s = env.reset()


    def step(self, a):
        """TODO: Docstring for function.

        Parameters
        ----------
        arg1 : action

        Returns
        -------
        TODO

        """
        return env.step(a)


    def reset(self):
        """TODO: Docstring for reset.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
        s0 = env.reset()
        self._s = s0
        return s0


class MDPR(MDP):
    """Docstring for MDP.

    assuming we have simualtor

    """

    def __init__(self, env, T, R):
        """TODO: to be defined1.

        assuming we have openai env

        Parameters
        ----------
        env : TODO

        """
        self._env = env
        self._s = env.reset()


    def step(self, a):
        """TODO: Docstring for function.

        Parameters
        ----------
        arg1 : action

        Returns
        -------
        TODO

        """
        s_next, r, done, _ =  env.step(a)
        return s_next, self._R(s, a), done, _


    def reset(self):
        """TODO: Docstring for reset.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
        s0 = env.reset()
        self._s = s0
        return s0

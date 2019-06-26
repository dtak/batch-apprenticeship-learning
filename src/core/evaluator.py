import numpy as np
import itertools


class ALEvaluator(object):
    """

    Evaluator for various Apprenticeship Learnign algorithms

    currently supports

    - perf : E[V^pi_eval(s0)] / E[V^pi_exp(s0)]
    - a_match : sum_D I(a_pi_eval == a_pi_exp) / len(D)

    """

    def __init__(self, D, gamma, env=None):
        """TODO: to be defined1.

        Parameters
        ----------
        D : TODO
        arg : TODO


        """
        self._D = D
        self._env = env
        self._gamma = gamma
        self._perf_exp = None

    @property
    def perf_exp(self):
        return self._perf_exp

    @perf_exp.setter
    def perf_exp(self, perf):
        self._perf_exp = perf


    def _evaluate_perf_onpolicy(self, pi, stochastic):
        """

        currently thisdule assumes access to the true env
        to validate policy on


        should support off-policy/batch evaluation

        Parameters
        ----------
        pi : TODO

        Returns
        -------
        TODO

        """
        sample_size = 100
        v_list = []

        for epi_i in range(sample_size):

            # this is not fixed
            s = self._env.reset()
            v = 0.0
            for t in itertools.count():
                a = pi.act(stochastic, s)[0]
                s_next, r, done, _ = self._env.step(a)
                v += self._gamma**t *r
                s = s_next
                if done:
                    break
            v_list.append(v)
        return np.mean(v_list)


    @staticmethod
    def evaluate_perf_env(pi, env, gamma, stochsatic):
        """TODO: Docstring for evaluate_perf.
        Returns
        -------
        TODO

        """
        sample_size = 100
        v_list = []

        for epi_i in range(sample_size):

            # this is not fixed
            s = env.reset()
            v = 0.0
            for t in itertools.count():
                a = pi.act(stochastic, s)[0]
                s_next, r, done, _ = env.step(a)
                v += gamma**t *r
                s = s_next
                if done:
                    break
            v_list.append(v)
        return np.mean(v_list)



    def evaluate_perf(self, pi, stochastic):
        """

        currently this module assumes access to the true env
        to validate policy on


        should support off-policy/batch evaluation

        Parameters
        ----------
        pi : TODO

        Returns
        -------
        TODO

        """
        perf = self._evaluate_perf_onpolicy(pi, stochastic)
        return perf / self._perf_exp


    def evaluate_a_match(self, pi, stochastic):
        D_s = self._D["s"]
        D_a = self._D["a"]

        matched = 0
        for s, a in zip(D_s, D_a):
            try:
                matched += int(a == pi.act(stochastic, s)[0])
            except:
                matched += int(a == pi.act(stochastic, s[np.newaxis,...])[0])


        return matched / len(D_s)




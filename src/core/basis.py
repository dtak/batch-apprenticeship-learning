import numpy as np
from numpy.linalg import norm
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import sklearn.preprocessing


class RBFKernel(object):
    """Docstring for RBFKernel. """

    def __init__(self, env, n_component=25, gammas=[1.0], include_action_to_basis=False, include_action=False):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        n_component : TODO, optional
        """
        states = np.array([env.observation_space.sample() for x in range(10000)])
        actions = np.array([env.action_space.sample() for x in range(10000)]).reshape(10000, 1)
        # giving state action
        if include_action_to_basis:
            xs = np.hstack((states, actions))
        else:
            # giving state
            xs = states
        self._include_action = include_action
        self._include_action_to_basis = include_action_to_basis

        scaler = sklearn.preprocessing.StandardScaler()

        scaler.fit(xs)
        self._scaler = scaler
        self._n_component = n_component
        self._gammas = gammas
        self._phi = self.fit(scaler.transform(xs))


    def fit(self, scaled):
        feature_list = []
        for i, g in enumerate(self._gammas):
            f = ("rbf{}".format(i), RBFSampler(gamma=g, n_components=self._n_component))
            feature_list.append(f)
        phi = sklearn.pipeline.FeatureUnion(feature_list)
        phi.fit(scaled)
        return phi


    def transform(self, s, a):
        """
        """
        # giving state action
        if self._include_action_to_basis:
            sa = np.hstack((s, a))
            if len(sa.shape) == 1:
                sa = np.expand_dims(sa, axis=0)
            x = self._scaler.transform(sa)
            featurized = self._phi.transform(x)
            return featurized
        elif self._include_action:
            if len(s.shape) == 1:
                s = np.expand_dims(s, axis=0)
            x = self._scaler.transform(s)
            featurized = self._phi.transform(x)
            return np.expand_dims(np.hstack((featurized[0], a)), axis=0)
        else:
            if len(s.shape) == 1:
                s = np.expand_dims(s, axis=0)
            x = self._scaler.transform(s)
            featurized = self._phi.transform(x)
            return featurized


class RBFKernel2(object):
    """Docstring for RBFKernel. """

    def __init__(self, states, n_action, p, components=[25], gammas=[1.0], include_action=False):
        """TODO: to be defined1.

        assume action is discrete

        Returns: n x (|A| x k) feature matrix

        Parameters
        ----------
        env : TODO
        n_component : TODO, optional
        """
        # giving state action
        self._include_action = include_action

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(states)
        self._scaler = scaler

        self._components = components
        self._gammas = gammas
        self._n_action = n_action
        self._p = p
        self._phi = self.fit(scaler.transform(states))


    def fit(self, scaled):
        feature_list = []
        for i, (g, c) in enumerate(zip(self._gammas, self._components)):
            f = ("rbf{}".format(i), RBFSampler(gamma=g, n_components=c))
            feature_list.append(f)
        phi = sklearn.pipeline.FeatureUnion(feature_list)
        phi.fit(scaled)
        return phi


    def transform(self, s, a):
        """
        """
        def helper(s, a):
            featurized = np.zeros(self._p, dtype=np.float)

            l = int(self._p*a/self._n_action)
            r = int(self._p*(a+1)/self._n_action)
            featurized[l:r] = np.array(s)
            return featurized

        def helper_batch(x):
            phi_s, a = x[:-1], x[-1]
            a = a.astype(np.int)
            featurized = np.zeros(self._p, dtype=np.float)
            l = int(self._p*a/self._n_action)
            r = int(self._p*(a+1)/self._n_action)
            featurized[l:r] = phi_s
            return featurized

        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)
        s = self._scaler.transform(s)
        phi_s = self._phi.transform(s)

        if s.shape[0] == 1:
            phi_sa = np.expand_dims(helper(phi_s, a), axis=0)
        else:
            if len(a.shape) == 1:
                a = np.vstack(a)
            phi_sa = np.apply_along_axis(helper_batch, 1, np.hstack((phi_s, a)))

        assert phi_sa.shape == (s.shape[0], self._p)
        return phi_sa


def get_linear_basis(include_action=False):
    def f(s, a):
        if include_action:
            sa = np.hstack((s, a))
            if len(sa.shape) == 1:
                sa = np.expand_dims(sa, axis=0)
            return sa
        else:
            if len(s.shape) == 1:
                s = np.expand_dims(s, axis=0)
            return s
    return f


class LinearKernel2(object):
    """Docstring for LinearBasis. """

    def __init__(self, p, n_action, include_action=False):
        """TODO: to be defined1.

        Parameters
        ----------
        p : TODO
        n_action : TODO
        include_action : TODO, optional


        """
        self._p = p
        self._n_action = n_action
        self._include_action = include_action

    def transform(self, s, a):
        """TODO: Docstring for transform.

        assume action is discrete

        Returns: n x (|A| x k) feature matrix

        Parameters
        ----------
        s : TODO
        a : TODO

        Returns
        -------
        TODO

        """


        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)

        if s.shape[0] == 1:
            phi_sa = np.expand_dims(self._helper(s, a), axis=0)
        else:
            phi_sa = np.apply_along_axis(self._helper_batch, 1, np.hstack((s, a)))
        assert phi_sa.shape == (s.shape[0], self._p)
        return phi_sa

    def _helper(self, s, a):
        featurized = np.zeros(self._p, dtype=np.float)
        l = int(self._p*a/self._n_action)
        r = int(self._p*(a+1)/self._n_action)
        featurized[l:r] = np.array(s)
        return featurized


    def _helper_batch(self, x):
        phi_s, a = x[:-1], x[-1]
        a = a.astype(np.int)
        featurized = np.zeros(self._p, dtype=np.float)
        l = int(self._p*a/self._n_action)
        r = int(self._p*(a+1)/self._n_action)
        featurized[l:r] = phi_s
        return featurized


class GaussianKernel(object):
    """Docstring for RBFKernel. """

    def __init__(self, states,
                       n_action,
                       p,
                       n_component,
                       include_action=False,
                       add_bias=False,
                       scaler=None,
                       standardized=False):
        """

        currently only work for 2d 

        assume action is discrete

        Returns: n x (|A| x k) feature matrix

        Parameters
        ----------
        env : TODO
        n_component : TODO, optional
        """
        # giving state action
        self._include_action = include_action
        self._n_component = n_component
        self._add_bias = add_bias
        self._n_action = n_action
        self._p = p


        self._standardized = standardized

        if self._standardized:
            self._scaler = scaler
            self._phi = self.fit(scaler.transform(states))
        else:
            self._phi = self.fit(states)


    def fit(self, states):
        if self._standardized:
            #raise Exception("Does not work yet.")
            x, y = states[:, 0], states[:, 1]
            c = 1.1
            a = np.linspace(c * x.min(), c * x.max(), self._n_component)
            b = np.linspace(c * y.min(), c * y.max(), self._n_component)
            self._mu_x, self._mu_y = np.meshgrid(a, b)
            # since standardized
            self._sig = 1.0
        else:
            a = np.linspace(-1.2, 0.6, self._n_component)
            b = np.linspace(-0.07, 0.07, self._n_component)
            self._mu_x, self._mu_y = np.meshgrid(a, b)
            self._sig_x = 2*np.power((0.6+1.2)/10.,2)
            self._sig_y = 2*np.power((0.07+0.07)/10.,2)


        mus = np.vstack(([self._mu_x.T], [self._mu_y.T])).T
        self._mus = mus.reshape(self._n_component**2, 2)

        return self._phi


    def _phi(self, s):
        self._s = s.flatten()
        gk_vec = np.vectorize(self._gauss_kernel)
        phi = gk_vec(self._mu_x, self._mu_y)
        if self._add_bias:
            phi = np.append(phi, [1.])
        return phi

    def _gauss_kernel(self, mu_x, mu_y):
        if self._standardized:
            return np.exp(-norm(self._s - np.array([mu_x, mu_y]), 2)**2/(2*self._sig**2))
        else:
            pos, speed = self._s
            a = np.power(pos-mu_x,2)/self._sig_x
            b = np.power(speed-mu_y,2)/self._sig_y
            return np.exp(-a - b)


    def transform_s(self, s):
        """TODO: Docstring for transform_a.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """

        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)

        if self._standardized:
            s = self._scaler.transform(s)

        phi_s = np.apply_along_axis(self._phi, 1, s)
        assert phi_s.shape == (s.shape[0], int(self._p / self._n_action))
        return phi_s


    def transform(self, s, a):
        """
        """
        def helper(s, a):
            featurized = np.zeros(self._p, dtype=np.float)

            l = int(self._p*a/self._n_action)
            r = int(self._p*(a+1)/self._n_action)
            featurized[l:r] = self._phi(s)
            return featurized

        def helper_batch(x):
            s, a = x[:-1], x[-1]
            a = a.astype(np.int)
            featurized = np.zeros(self._p, dtype=np.float)
            l = int(self._p*a/self._n_action)
            r = int(self._p*(a+1)/self._n_action)
            featurized[l:r] = self._phi(s)
            return featurized

        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)

        if self._standardized:
            s = self._scaler.transform(s)

        if s.shape[0] == 1:
            phi_sa = np.expand_dims(helper(s, a), axis=0)
        else:
            if len(a.shape) == 1:
                a = np.vstack(a)
            phi_sa = np.apply_along_axis(helper_batch, 1, np.hstack((s, a)))

        assert phi_sa.shape == (s.shape[0], self._p)
        #print("phi_sa:{}".format(phi_sa[0, :, :]))
        return phi_sa

    def transform_a_next(self, phi_s_next, a):
        """TODO: Docstring for phi_sa.



        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """

        def helper_batch(x):
            phi_s_next, a = x[:-1], x[-1]
            featurized = np.zeros(self._p, dtype=np.float)
            l = int(self._p*a/self._n_action)
            r = int(self._p*(a+1)/self._n_action)
            featurized[l:r] = phi_s_next
            return featurized

        if len(a.shape) == 1:
            a = np.vstack(a)
        phi_sa_conc = np.hstack((phi_s_next, a))
        phi_sa_next = np.apply_along_axis(helper_batch, 1, phi_sa_conc)
        assert phi_sa_next.shape == (phi_s_next.shape[0], self._p)
        return phi_sa_next


class IdentityBasis(object):
    """Docstring for BasisFunction

    Neutral Basis function for batch
    likely used for psi for feature expectation

    """

    def __init__(self, q, n_action, scaler=None, standardized=True):
        """

        Parameters
        ----------
        q : dimension of features
        n_action : number of actions

        Returns
        -------
        TODO

        """
        self._q = q
        self._n_action = n_action
        self._scaler = scaler
        self._standardized = standardized


    def transform(self, s, a):
        """ s x a -> [0 ... s ...0]

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        transformed

        """

        def helper(x):
            s, a = x[:-1], x[-1]
            a = a.astype(np.int)
            featurized = np.zeros(self._q, dtype=np.float)
            l = int(self._q*a/self._n_action)
            r = int(self._q*(a+1)/self._n_action)
            featurized[l:r] = s
            return featurized


        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)


        if np.isscalar(a):
            a = np.expand_dims([a], axis=1)

        if len(a.shape) == 1:
            a = np.expand_dims(a, axis=1)

        if self._standardized:
            s = self._scaler.transform(s)


        transformed = np.apply_along_axis(helper, 1, np.hstack((s, a)))
        assert transformed.shape == (s.shape[0], self._q)
        return transformed



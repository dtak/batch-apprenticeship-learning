'''
heavily modified. 

the original implementation credit: openai baselines
'''
import tensorflow as tf
import gym

#import baselines.common.tf_util as U
#from baselines.common.mpi_running_mean_std import RunningMeanStd
#from baselines.common.distributions import make_pdtype
#from baselines.acktr.utils import dense

import contrib.baselines.common.tf_util as U
from contrib.baselines.common.mpi_running_mean_std import RunningMeanStd
from contrib.baselines.common.distributions import make_pdtype
from contrib.baselines.acktr.utils import dense


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size_phi, num_hid_layers_phi,
            dim_phi, gaussian_fixed_var=True):
        """

        input: ob, T_ac established as placeholder

        output: ac, ob_next

        """
        assert isinstance(ob_space, gym.spaces.Box)
        self._ob_space = ob_space

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        # normalize obs and clip them
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz

        # define phi (shared parameter for useful representation)
        #for i in range(num_hid_layers_phi):
        hid_size_list = [hid_size_phi] * num_hid_layers_phi + [dim_phi]
        for i, hid_size in enumerate(hid_size_list):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "phi%i" % (i+1), weight_init=U.normc_initializer(1.0)))


        self.phi = phi = last_out
        self._featurize = U.function([ob], [phi])

        # define v^pi
        self.vpred = dense(phi, 1, "vf_final", weight_init=U.normc_initializer(1.0))[:, 0]

        # define pi(a|s)
        if  gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            # continuous action
            mean = dense(phi, pdtype.param_shape()[0]//2, "pi_final_mu", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="pi_final_logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            # discrete action
            pdparam = dense(phi, pdtype.param_shape()[0], "pi_final", U.normc_initializer(0.01))
        self.pi = pi = pdtype.pdfromflat(pdparam)

        pi_stochastic = U.get_placeholder(name="pi_stochastic", dtype=tf.bool, shape=())
        self.ac = ac = U.switch(pi_stochastic, pi.sample(), pi.mode())
        self._act = U.function([pi_stochastic, ob], [ac, self.vpred, pi.logits])

        # define T(s'|s, a) and a~pi(a|s)
        self.ob_next_pdtype = ob_next_pdtype = make_pdtype(ob_space)

        if isinstance(ac_space, gym.spaces.Box):
            # if continuous action
            T_ac = U.get_placeholder(name="T_ac", dtype=tf.float32,
                    shape=[sequence_length] + [pdtype.param_shape()[0]//2])
        else:
            # if discrete action
            T_ac = U.get_placeholder(name="T_ac", dtype=tf.float32,
                    shape=[sequence_length] + list(ac_space.shape))

        T_input = tf.concat([phi, tf.expand_dims(T_ac, 1)], axis=1)
        T_mean = dense(T_input, ob_space.shape[0], "T_final_mu", U.normc_initializer(0.01))
        T_logstd = tf.get_variable(name="T_final_logstd", shape=[1,
            ob_space.shape[0]], initializer=tf.zeros_initializer())
        T_pdparam = tf.concat([T_mean, T_mean * 0.0 + T_logstd], axis=1)

        self.T = T = ob_next_pdtype.pdfromflat(T_pdparam)
        T_stochastic = U.get_placeholder(name="T_stochastic", dtype=tf.bool, shape=())
        self.ob_next = ob_next = U.switch(T_stochastic, T.sample(), T.mode())
        self._predict_ob_next = U.function([T_stochastic, ob, T_ac], [ob_next])


    def act(self, stochastic, ob):
        ac1, vpred1, p1 = self._act(stochastic, ob)
        return ac1, vpred1, p1

    def predict_ob_next(self, stochastic, ob, ac):
        ob_next = self._predict_ob_next(stochastic, ob[None], ac[None])
        return ob_next[0]

    def featurize(self, ob):
        if len(ob.shape) == len(self._ob_space.shape):
            # one sample (X,) -> (1, X)
            ob = ob[None]
        phi = self._featurize(ob)
        return phi[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


class MlpPolicyMultiGaussian(object):
    recurrent = False

    def __init__(self, *args, **kwargs):
        self.scope = tf.get_variable_scope().name
        self._init(*args, **kwargs)
        self.mu_func

    #def __init__(self, name, reuse=False, *args, **kwargs):
    #    with tf.variable_scope(name):
    #        if reuse:
    #            tf.get_variable_scope().reuse_variables()
    #        self._init(*args, **kwargs)
    #        self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, mu_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self._ob_space = ob_space

        self.ac_pdtype = ac_pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="obs_t", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))


        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        # define mu ~ mu_est(s, a)
        self.mu_pdtype = mu_pdtype = make_pdtype(mu_space)

        if isinstance(ac_space, gym.spaces.Box):
            # if continuous action
            mu_ac = U.get_placeholder(name="act_t", dtype=tf.float32,
                    shape=[sequence_length] + [ac_pdtype.param_shape()[0]//2])
        else:
            # if discrete action
            mu_ac = U.get_placeholder(name="act_t", dtype=tf.float32,
                    shape=[sequence_length] + list(ac_space.shape))

        ob_ac_input = tf.concat([ob, tf.expand_dims(mu_ac, 1)], axis=1, name="mu_input")
        mu_mean = dense(ob_ac_input, mu_space.shape[0], "mu_final_mean", U.normc_initializer(0.01))
        mu_logstd = tf.get_variable(name="mu_final_logstd", shape=[1,
            mu_space.shape[0]], initializer=tf.zeros_initializer())
        mu_pdparam = tf.concat([mu_mean, mu_mean * 0.0 + mu_logstd], axis=1)

        self.mu = mu = mu_pdtype.pdfromflat(mu_pdparam)
        mu_stochastic = U.get_placeholder(name="mu_stochastic", dtype=tf.bool, shape=())
        self.mu_est = mu_est = U.switch(mu_stochastic, mu.sample(), mu.mode())

        self.mu_func = lambda mu_stochastic, ob, mu_ac : mu_est



    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


class MlpPolicyOriginal(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i+1), weight_init=U.normc_initializer(1.0)))

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.vpred, self.pd.logits])

    def act(self, stochastic, ob):
        ac1, vpred1, p1 = self._act(stochastic, ob)
        return ac1, vpred1, p1

    def act1(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []



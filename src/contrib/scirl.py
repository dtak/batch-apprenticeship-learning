"""

credit: https://github.com/edouardklein/RL-and-IRL

adapted.

"""
#from pylab import *

def non_scalar_vectorize(func, input_shape, output_shape):
    """Return a featurized version of func, where func takes a potentially matricial argument and returns a potentially matricial answer.

    These functions can not be naively vectorized by numpy's vectorize.

    With vfunc = non_scalar_vectorize( func, (2,), (10,1) ),

    func([p,s]) will return a 2D matrix of shape (10,1).

    func([[p1,s1],...,[pn,sn]]) will return a 3D matrix of shape (n,10,1).

    And so on.
    """
    def vectorized_func(arg):
        #print 'Vectorized : arg = '+str(arg)
        nbinputs = prod(arg.shape)/prod(input_shape)
        if nbinputs == 1:
            return func(arg)
        outer_shape = arg.shape[:len(arg.shape)-len(input_shape)]
        outer_shape = outer_shape if outer_shape else (1,)
        arg = arg.reshape((nbinputs,)+input_shape)
        answers=[]
        for input_matrix in arg:
            answers.append(func(input_matrix))
        return array(answers).reshape(outer_shape+output_shape)
    return vectorized_func

def zip_stack(*args):
    """Given matrices of same shape, return a matrix whose elements are tuples from the arguments (i.e. with one more dimension).

    zip_stacking three matrices of shape (n,p) will yeld a matrix of shape (n,p,3)
    """
    shape = args[0].shape
    nargs = len(args)
    args = [m.reshape(-1) for m in args]
    return array([el for el in zip(*args)]).reshape(shape+(nargs,))

def f_geq(a,b):
    r"Float inequality, return True if $a\geq b-\epsilon$"
    return a - b > -1e-10
def f_eq(a,b):
    r"Float equality, return True if $|a-b| < \epsilon$"
    return abs( a-b ) < 1e-10


# Quick and dirty implementation of LSPI
# http://catbert.cs.duke.edu/~parr/jmlr03.pdf

#from pylab import *
#from stuff import *

GAMMA=0.9 #Discout factor
LAMBDA=0#.1 #Regularization coeff for LSTDQ

def greedy_policy( omega, phi, A, s_dim=2 ):
    def policy( *args ):
        state_actions = [hstack(args+(a,)) for a in A]
        q_value = lambda sa: float(dot(omega.transpose(),phi(sa)))
        best_action = argmax( state_actions, q_value )[-1] #FIXME6: does not work for multi dimensional actions
        return best_action
    vpolicy = non_scalar_vectorize( policy, (s_dim,), (1,1) )
    return lambda state: vpolicy(state).reshape(state.shape[:-1]+(1,))

def lstdq(phi_sa, phi_sa_dash, rewards, phi_dim=1):
    #print "shapes of phi de sa, phi de sprim a prim, rewards"+str(phi_sa.shape)+str(phi_sa_dash.shape)+str(rewards.shape)
    A = zeros((phi_dim,phi_dim))
    b = zeros((phi_dim,1))
    for phi_t,phi_t_dash,reward in zip(phi_sa,phi_sa_dash,rewards):
        A = A + dot( phi_t,
                     (phi_t - GAMMA*phi_t_dash).transpose())
        b = b + phi_t*reward
    return dot(inv(A + LAMBDA*identity( phi_dim )),b)

def lspi( data, s_dim=1, a_dim=1, A = [0], phi=None, phi_dim=1, epsilon=0.01, iterations_max=30,
          plot_func=None):
    nb_iterations=0
    sa = data[:,0:s_dim+a_dim]
    phi_sa = phi(sa)
    s_dash = data[:,s_dim+a_dim:s_dim+a_dim+s_dim]
    rewards = data[:,s_dim+a_dim+s_dim]
    omega = zeros(( phi_dim, 1 ))
    #omega = genfromtxt("../Code/InvertedPendulum/omega_E.mat").reshape(30,1)
    diff = float("inf")
    cont = True
    policy = greedy_policy( omega, phi, A )
    while cont:
        if plot_func:
            plot_func(omega)
        sa_dash = hstack([s_dash,policy(s_dash)])
        phi_sa_dash = phi(sa_dash)
        omega_dash = lstdq(phi_sa, phi_sa_dash, rewards, phi_dim=phi_dim)
        diff = norm( omega_dash-omega )
        omega = omega_dash
        policy = greedy_policy( omega, phi, A )
        nb_iterations+=1
        print("LSPI, iter :"+str(nb_iterations)+", diff : "+str(diff))
        if nb_iterations > iterations_max or diff < epsilon:
            cont = False
    sa_dash = hstack([s_dash,policy(s_dash)])
    phi_sa_dash = phi(sa_dash)
    omega = lstdq(phi_sa, phi_sa_dash, rewards, phi_dim=phi_dim) #Omega is the Qvalue of pi, but pi is not the greedy policy w.r.t. omega
    return policy,omega


def argmax( set, func ):
     return max( zip( set, map(func,set) ), key=lambda x:x[1] )[0]



# This is an exemple of SCIRL running on the mountain car toy problem.
# Mountain car : http://en.wikipedia.org/wiki/Mountain_Car
# SCIRL : http://rdklein.fr/research/papers/klein2012structured.pdf
# LSTD-mu : http://rdklein.fr/research/papers/klein2011batch.pdf
# Structured Classification : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.6014&rep=rep1&type=pdf


#from pylab import *
#from random import *
#import numpy
#from stuff import * #Some utility functions
#from rl import * #Reinforcement learning code

# First, the generic code for SCIRL. SCIRL is basically a structured classifier whose features are the feature expectation of the expert
# This structured classifier is basically a gradient descent
# So, firstly first, a simple gradient descent implementation. Its need the gradient function and the step size parameter.
class GradientDescent(object):

   def alpha( self, t ):
      raise NotImplementedError("Cannot call abstract method")

   theta_0=None
   Threshold=None
   T = -1
   sign = None

   def run( self, f_grad, f_proj=None, b_norm=False ): #grad is a function of theta
      theta = self.theta_0.copy()
      best_theta = theta.copy()
      best_norm = float("inf")
      best_iter = 0
      t=0
      while True:#Do...while loop
         t+=1
         DeltaTheta = f_grad( theta )
         current_norm = norm( DeltaTheta )
         if b_norm and  current_norm > 0.:
             DeltaTheta /= norm( DeltaTheta )
         theta = theta + self.sign * self.alpha( t )*DeltaTheta
         if f_proj:
             theta = f_proj( theta )
         print("Gradient norm : "+str(current_norm)+", step : "+str(self.alpha(t))+", iteration : "+str(t))

         if current_norm < best_norm:
             best_norm = current_norm
             best_theta = theta.copy()
             best_iter = t
         if current_norm < self.Threshold or (self.T != -1 and t >= self.T):
             break

      print("Gradient norm : "+str(best_norm)+", iteration : "+str(best_iter))
      return best_theta

# From this gradient descent, the classification algorithm :
# There is an optimized version for sparse-ish matrices somewhere, if need be.
class StructuredClassifier(GradientDescent):
    sign=-1.
    Threshold=0.1 #Sensible default
    T=40 #Sensible default
    phi=None
    phi_xy=None
    inputs=None
    labels=None
    label_set=None
    dic_data={}
    x_dim=None

    def alpha(self, t):
        return 3./(t+1)#Sensible default

    def __init__(self, data, x_dim, phi, phi_dim, Y):
        self.x_dim=x_dim
        self.inputs = data[:,:-1]
        shape = list(data.shape)
        shape[-1] = 1
        self.labels = data[:,-1].reshape(shape)
        self.phi=phi
        self.label_set = Y
        self.theta_0 = zeros((phi_dim,1))
        self.phi_xy = self.phi(data)
        for x,y in zip(self.inputs,self.labels):
            self.dic_data[str(x)] = y

    def structure(self, xy):
        return 0. if xy[-1] == self.dic_data[str(xy[:-1])] else 1.

    def structured_decision(self, theta):
        def decision( x ):
            score = lambda xy: dot(theta.transpose(),self.phi(xy)) + self.structure(xy)
            input_label_couples = [hstack([x,y]) for y in self.label_set]
            best_label = argmax(input_label_couples, score)[-1]
            return best_label
        vdecision = non_scalar_vectorize(decision, (self.x_dim,), (1,1))
        return lambda x: vdecision(x).reshape(x.shape[:-1]+(1,))

    def gradient(self, theta):
        classif_rule = self.structured_decision(theta)
        y_star = classif_rule(self.inputs)
        #print "Gradient : "+str(y_star)
        #print str(self.labels)
        phi_star = self.phi(hstack([self.inputs,y_star]))
        return mean(phi_star-self.phi_xy,axis=0)

    def run(self):
        f_grad = lambda theta: self.gradient(theta)
        theta = super(StructuredClassifier,self).run( f_grad, b_norm=True)
        classif_rule = greedy_policy(theta,self.phi,self.label_set)
        return classif_rule,theta



import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

def train_scirl(D, phi, evaluator):
    """
    a simplified variant of the original implementation
    may prevent the low-data regime problem
    """
    s = D["s"]
    a = D["a"]
    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=1)
    s_next = D["s_next"]
    done = D["done"]
    if len(done.shape) == 1:
        done = np.expand_dims(done, axis=1)
    phi_sa = D["phi_sa"]

    n_transition = D["s"].shape[0]
    idx = idx = int(n_transition * 0.7)

    #D_train = {s[:idx, :],
    #     a[:idx, :],
    #     phi_sa[:idx, :],
    #     s_next[:idx, :],
    #     done[:idx, :]

    D_train = {"s" : s[:idx, :],
             "a" : a[:idx, :],
             "phi_sa" : phi_sa[:idx, :],
             "s_next": s_next[:idx, :],
             "done": done[:idx, :]}

    D_val = {"s" : s[idx:, :],
           "a" : a[idx:, :],
           "phi_sa" : phi_sa[idx:, :],
           "s_next": s_next[idx:, :],
           "done": done[idx:, :]}


    # train theta_c using MC^2
    Phi_s = phi(D_train["s"])
    y_a = D_train["a"]

    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(Phi_s, y_a)

    def reward_fn(s):
        return np.dot(clf.coef_, phi(s)) + clf.intercept_


    def make_policy(clf):

        def policy(s):
            # decision
            # deterministic
            return clf.predict(phi(s))
        return policy


    policy = make_policy(clf)


    D_val_s = D_val["s"]
    D_val_a = D_val["a"]
    matched = 0
    for s, a in zip(D_val_s, D_val_a):
        matched += int(a == policy(s)[0])

    a_match =  matched / len(D_val_s)



    results = {
            "a_match" : a_match,
            "theta_c_coef": clf.coef_,
            "theta_c_intercept": clf.intercept_,
            "reward_fn": reward_fn,
            "policy": policy,
            }
    return results

def train_scirl_v2(D, phi, evaluator, phi_dim, task_desc, params):
    """
    close to original implementation
    under-determined for low-data regime
    """

    s = D["s"]
    a = D["a"]
    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=1)
    s_next = D["s_next"]
    done = D["done"]
    if len(done.shape) == 1:
        done = np.expand_dims(done, axis=1)
    phi_sa = D["phi_sa"]

    n_transition = D["s"].shape[0]
    idx = idx = int(n_transition * 0.7)

    #D_train = {s[:idx, :],
    #     a[:idx, :],
    #     phi_sa[:idx, :],
    #     s_next[:idx, :],
    #     done[:idx, :]

    D_train = {"s" : s[:idx, :],
             "a" : a[:idx, :],
             "phi_sa" : phi_sa[:idx, :],
             "s_next": s_next[:idx, :],
             "done": done[:idx, :]}

    D_val = {"s" : s[idx:, :],
           "a" : a[idx:, :],
           "phi_sa" : phi_sa[idx:, :],
           "s_next": s_next[idx:, :],
           "done": done[idx:, :]}


    n_action = task_desc["n_action"]
    eps = params["eps"]
    gamma =  task_desc["gamma"]
    p = q = phi_dim # adding action dim
    stochastic = True

    from core.feature_expectation import EmpiricalMuEstimator, LSTDMuEstimator
    from core.policy import EmpiricalPolicy



    indices = np.random.choice(range(D["s"].shape[0]), size=10)
    pi_eval = EmpiricalPolicy(D, n_action)
    mu_estimator = LSTDMuEstimator(phi, gamma, D, p, q, eps, D["s"][indices, :])
    mu_estimator.fit(pi_eval, stochastic)


    def reward_fn(s):
        return np.dot(theta_SCIRL, phi(s))


    def make_policy(theta):
        def policy(s):
            # decision
            # deterministic

            score_sa_list = []

            for a in range(n_action):
                mu_est_sa = mu_estimator.estimate(s[None], a)
                score_sa = np.dot(theta, mu_est_sa)
                score_sa_list.append(score_sa)

            return np.argmax(score_sa_list)
        return policy

    max_steps = 100
    np.random.seed(1)
    theta = np.random.rand(phi_dim)
    policy = make_policy(theta)
    alpha = 0.01

    X = D_train["s"]
    y = D_train["a"]
    n_sample = X.shape[0]
    idx = 0
    t = 0
    minibatch_size = 32

    while t < max_steps:

        idx_p1 = np.max([idx + minibatch_size, n_action])
        B_x = X[idx : idx_p1]
        B_y = y[idx : idx_p1]
        idx += minibatch_size

        if idx >= n_sample:
            # add shuffle
            idx = 0

        a_predict = np.array([policy(s) for s in B_x])
        phi_est = mu_estimator.estimate(B_x, a_predict)
        phi_sa = mu_estimator.estimate(B_x, B_y)
        grad = np.mean(phi_est-phi_sa, axis=0)

        theta += alpha * grad
        t += 1


    D_val_s = D_val["s"]
    D_val_a = D_val["a"]
    matched = 0
    for s, a in zip(D_val_s, D_val_a):
        matched += int(a == policy(s))

    a_match =  matched / len(D_val_s)


    results = {
            "a_match" : a_match,
            "theta_c": theta_SCIRL,
            "reward_fn": reward_fn,
            "policy": policy,
            }
    return results

def train_scirl_v3(D, phi, evaluator):
    """needs fixing
    """
    s = D["s"]
    a = D["a"]
    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=1)
    s_next = D["s_next"]
    done = D["done"]
    if len(done.shape) == 1:
        done = np.expand_dims(done, axis=1)
    phi_sa = D["phi_sa"]

    n_transition = D["s"].shape[0]
    idx = idx = int(n_transition * 0.7)

    #D_train = {s[:idx, :],
    #     a[:idx, :],
    #     phi_sa[:idx, :],
    #     s_next[:idx, :],
    #     done[:idx, :]

    D_train = {"s" : s[:idx, :],
             "a" : a[:idx, :],
             "phi_sa" : phi_sa[:idx, :],
             "s_next": s_next[:idx, :],
             "done": done[:idx, :]}

    D_val = {"s" : s[idx:, :],
           "a" : a[idx:, :],
           "phi_sa" : phi_sa[idx:, :],
           "s_next": s_next[idx:, :],
           "done": done[idx:, :]}

    single_mu = lambda sa:feature_expectations[str(sa)]
    mu_E = non_scalar_vectorize(single_mu, (3,), (50,1))
    SCIRL = StructuredClassifier(sa, 2, mu_E, 50, ACTION_SPACE)
    void,theta_SCIRL = SCIRL.run()

    def reward_fn(s):
        return np.dot(theta_SCIRL, phi(s,a))

    def policy(s):
        # decision
        # deterministic
        return np.argmax(np.dot(theta_SCIRL, phi(s)))


    D_val_s = D_val["s"]
    D_val_a = D_val["a"]
    matched = 0
    for s, a in zip(D_val_s, D_val_a):
        matched += int(a == policy(s)[0])

    a_match =  matched / len(D_val_s)



    results = {
            "a_match" : a_match,
            "theta_c": theta_SCIRL,
            "reward_fn": reward_fn,
            "policy": policy,
            }
    return results

def train_scirl_v4(D, phi, evaluator):
    """needs fixing
    """
    s = D["s"]
    a = D["a"]
    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=1)
    s_next = D["s_next"]
    done = D["done"]
    if len(done.shape) == 1:
        done = np.expand_dims(done, axis=1)
    phi_sa = D["phi_sa"]

    n_transition = D["s"].shape[0]
    idx = idx = int(n_transition * 0.7)

    #D_train = {s[:idx, :],
    #     a[:idx, :],
    #     phi_sa[:idx, :],
    #     s_next[:idx, :],
    #     done[:idx, :]

    D_train = {"s" : s[:idx, :],
             "a" : a[:idx, :],
             "phi_sa" : phi_sa[:idx, :],
             "s_next": s_next[:idx, :],
             "done": done[:idx, :]}

    D_val = {"s" : s[idx:, :],
           "a" : a[idx:, :],
           "phi_sa" : phi_sa[idx:, :],
           "s_next": s_next[idx:, :],
           "done": done[idx:, :]}


    single_mu = lambda sa:feature_expectations_MC[str(sa)]
    mu_E = non_scalar_vectorize(single_mu, (3,), (50,1))
    SCIRL_MC = StructuredClassifier(sa, 2, mu_E, 50, ACTION_SPACE)
    void,theta_SCIRL_MC = SCIRL_MC.run()
    SCIRL_reward_MC = lambda sas:dot(theta_SCIRL_MC.transpose(),psi(sas[:2]))[0]
    vSCIRL_reward_MC = non_scalar_vectorize( SCIRL_reward, (5,),(1,1) )


    def reward_fn(s):
        return np.dot(theta_SCIRL, phi(s,a))

    def policy(s):
        # decision
        # deterministic
        return np.argmax(np.dot(theta_SCIRL, phi(s)))


    D_val_s = D_val["s"]
    D_val_a = D_val["a"]
    matched = 0
    for s, a in zip(D_val_s, D_val_a):
        matched += int(a == policy(s)[0])

    a_match =  matched / len(D_val_s)



    results = {
            "a_match" : a_match,
            "theta_c": theta_SCIRL,
            "reward_fn": reward_fn,
            "policy": policy,
            }
    return results


# Now for the mountain car problem
# -1 :left, 0 : neutral, 1: right
ACTION_SPACE=[-1,0,1]
# The dynamics as explained on http://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar.html
def mountain_car_next_state(state,action):
    position,speed=state
    next_speed = squeeze(speed+action*0.001+cos(3*position)*(-0.0025))
    next_position = squeeze(position+next_speed)
    if not -0.07 <= next_speed <= 0.07:
        next_speed = sign(next_speed)*0.07
    if not -1.2 <= next_position <= 0.6:
        next_speed=0.
        next_position = -1.2 if next_position < -1.2 else 0.6
    return array([next_position,next_speed])


# The feature function over the state space is a Gaussian RBF with 7*7 nodes and a constant component :
#mountain_car_mu_position, mountain_car_mu_speed = meshgrid(linspace(-1.2,0.6,7),linspace(-0.07,0.07,7))

#mountain_car_sigma_position = 2*pow((0.6+1.2)/10.,2)
#mountain_car_sigma_speed = 2*pow((0.07+0.07)/10.,2)

#print("Shapes")
#print(mountain_car_mu_position.shape,mountain_car_mu_speed.shape)
#toto = zip_stack(mountain_car_mu_position, mountain_car_mu_speed)
def mountain_car_single_psi(state):
    position,speed=state
    psi=[]
    for mu in zip_stack(mountain_car_mu_position, mountain_car_mu_speed).reshape(7*7,2):
        psi.append(exp( -pow(position-mu[0],2)/mountain_car_sigma_position
                        -pow(speed-mu[1],2)/mountain_car_sigma_speed))
    psi.append(1.)
    return array(psi).reshape((7*7+1,1))

#mountain_car_psi= non_scalar_vectorize(mountain_car_single_psi,(2,),(50,1))

# The features over the state-action space :
# \phi(s,a) = \begin{pmatrix}\psi(s)\delta_{a=-1}\\\psi(s)\delta_{a=0}\\\psi(s)\delta_{a=1}\end{pmatrix}
# With \delta the Kronecker symbol

def mountain_car_single_phi(sa):
    state=sa[:2]
    index_action = int(sa[-1])+1
    answer=zeros(((7*7+1)*3,1))
    answer[index_action*(7*7+1):index_action*(7*7+1)+7*7+1] = mountain_car_single_psi(state)
    return answer

#mountain_car_phi= non_scalar_vectorize(mountain_car_single_phi,(3,),(150,1))


# The next two functions are a measure of how well the policy is controlling the car. The shorter the trajectory, the better
def mountain_car_episode_length(initial_position,initial_speed,policy):
    answer = 0
    reward = 0.
    state = array([initial_position,initial_speed])
    while answer < 300 and reward == 0. :
        action = policy(state)
        next_state = mountain_car_next_state(state,action)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        state=next_state
        answer+=1
    return answer

def mountain_car_episode_vlength(policy):
    return vectorize(lambda p,s:mountain_car_episode_length(p,s,policy))

# Various plotting functions
def mountain_car_plot( f, draw_contour=True, contour_levels=50, draw_surface=False ):
    '''Display a surface plot of function f over the state space'''
    pos = linspace(-1.2,0.6,30)
    speed = linspace(-0.07,0.07,30)
    pos,speed = meshgrid(pos,speed)
    Z = f(pos,speed)
    #fig = figure()
    if draw_surface:
        ax=Axes3D(fig)
        ax.plot_surface(pos,speed,Z)
    if draw_contour:
        contourf(pos,speed,Z,levels=linspace(min(Z.reshape(-1)),max(Z.reshape(-1)),contour_levels+1))
        colorbar()

def mountain_car_plot_policy( policy ):
    two_args_pol = lambda p,s:squeeze(policy(zip_stack(p,s)))
    mountain_car_plot(two_args_pol,contour_levels=3)

def mountain_car_V(omega):
    policy = greedy_policy( omega, mountain_car_phi, ACTION_SPACE )
    def V(pos,speed):
        actions = policy(zip_stack(pos,speed))
        Phi=mountain_car_phi(zip_stack(pos,speed,actions))
        return squeeze(dot(omega.transpose(),Phi))
    return V


# Now the IRL experiment setup
# This will be the expert policy
def mountain_car_manual_policy(state):
    position,speed = state
    return -1. if speed <=0 else 1.
# This is the standard reward for this problem
def mountain_car_reward(sas):
    position=sas[0]
    return 1 if position > 0.5 else 0
# The next three functions will sample trajectories from this policy, starting from the lower left quadrant of the state space
def mountain_car_interesting_state():
    position = numpy.random.uniform(low=-1.2,high=-0.9)
    speed = numpy.random.uniform(low=-0.07,high=0)
    return array([position,speed])

def mountain_car_IRL_traj():
    traj = []
    state = mountain_car_interesting_state()
    reward = 0
    while reward == 0:
        action = mountain_car_manual_policy(state)
        next_state = mountain_car_next_state(state, action)
        next_action = mountain_car_manual_policy(next_state)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        traj.append(hstack([state, action, next_state, next_action, reward]))
        state=next_state
    return array(traj)

def mountain_car_IRL_data(nbsamples):
    data = mountain_car_IRL_traj()
    while len(data) < nbsamples:
        data = vstack([data,mountain_car_IRL_traj()])
    return data[:nbsamples]

#TRAJS = mountain_car_IRL_data(100)
# This show the trajectories :
#scatter(TRAJS[:,0],TRAJS[:,1],c=TRAJS[:,2])
#axis([-1.2,0.6,-0.07,0.07])
#figure()
#s=TRAJS[:,:2]
#a=TRAJS[:,2]
#
#s_dash=TRAJS[:,3:5]
#a_dash=TRAJS[:,5]
#sa=TRAJS[:,:3]
#sa_dash=TRAJS[:,3:6]
#
#psi=mountain_car_psi
#phi=mountain_car_phi

# SCIRL is a structured classifier whose features are the feature expectation of the expert.
# When only using expert data, heuristics should be used, as described in the paper
# to compensate for lack of data s,a where a\neq \pi^E(s)
# we use two ways of computing mu_E, the expert's feature expectation
# one is to use LSTDmu, basically a vectorized version of LSTD
# Precomputing mu with LSTDmu and heuristics

## The other is to use a simple Monte-Carlo approach (with less and less data as we go forward in the trajectories)
##Precomputing mu with MC and heuristics
#feature_expectations_MC = {}
#for start_index in range(0,len(TRAJS)):
#    end_index = (i for i in range(start_index,len(TRAJS)) if TRAJS[i,6] == 1 or i==len(TRAJS)-1).__next__()
#    #print "start_index : "+str(start_index)+" end_index : "+str(end_index)
#    data_MC=TRAJS[start_index:end_index+1,:3]
#    GAMMAS = range(0,len(data_MC))
#    GAMMAS = array([el for el in map( lambda x: pow(GAMMA,x), GAMMAS)])
#    state_action = data_MC[0,:3]
#    state = data_MC[0,:2]
#    action = data_MC[0,2]
#    mu = None
#    if len(data_MC) > 1:
#        mu = dot( GAMMAS,squeeze(psi(data_MC[:,:2])))
#    else:
#        mu = squeeze(psi(squeeze(data_MC[:,:2])))
#    feature_expectations_MC[str(state_action)] = mu
#    for other_action in [a for a in ACTION_SPACE if a != action]:
#        state_action=hstack([state,other_action])
#        feature_expectations_MC[str(state_action)]=GAMMA*mu
#

# The starting state of training data for RL is not the same as for IRL
# For IRL we always start in the same quadrant
# For RL we start everywhere because we need the data to cover the whole dynamics
def mountain_car_uniform_state():
    return array([numpy.random.uniform(low=-1.2,high=0.6),numpy.random.uniform(low=-0.07,high=0.07)])

def mountain_car_training_data(freward=mountain_car_reward,traj_length=5,nb_traj=1000):
    traj = []
    random_policy = lambda s:choice(ACTION_SPACE)
    for i in range(0,nb_traj):
        state = mountain_car_uniform_state()
        reward=0
        t=0
        while t < traj_length and reward == 0:
            t+=1
            action = random_policy(state)
            next_state = mountain_car_next_state(state, action)
            reward = freward(hstack([state, action, next_state]))
            traj.append(hstack([state, action, next_state, reward]))
            state=next_state
    return array(traj)




#data = mountain_car_training_data(traj_length=5,nb_traj=1000) #In short : short trajectories of a random policy, starting randomly everywhere

# The data is of the form s a s r, with r the standard reward of the problem, which we are not supposed to know (plus, the policy we used as the expert is only near optimal for this reward)
# We replace the 'r' column with the reward from SCIRL, and train an agent with LSPI (for both flavor of SCIRL) :
#data[:,5] = squeeze(vSCIRL_reward(data[:,:5]))
#policy_SCIRL_LSTD,omega_SCIRL_LSTD = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=150, iterations_max=20 )

#data[:,5] = squeeze(vSCIRL_reward_MC(data[:,:5]))
#policy_SCIRL_MC,omega_SCIRL_MC = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=150, iterations_max=20 )

# Our measure of performance is the number of step needed to get out of the valley. The lesser the better.
# By modifying mountain_car_testing_state one can assess the quality on a more specific part of the state space
def mountain_car_testing_state():
    position = numpy.random.uniform(low=-1.2,high=0.5)
    speed = numpy.random.uniform(low=-0.07,high=0.07)
    return array([position,speed])

def mountain_car_mean_performance(policy):
    return mean([mountain_car_episode_length(state[0],state[1],policy) for state in [mountain_car_testing_state() for i in range(0,10)]])

#print("Performance for SCIRL, SCIRL_MC")
#print(mountain_car_mean_performance(policy_SCIRL_LSTD),mountain_car_mean_performance(policy_SCIRL_MC))

# From other experiments, classification alone averages around 120 steps
# This is not excellent, it flows from the fact that our trajectories all start
# from the same quadrant
# Due to the dynamics, the expert then visit the two upper quadrants of the state space
# but the lower right quadrant is never visited
# When evaluating a policy, we sometimes ask it to control the car in the lower right quadrant
# Classification is lost because it never has seen what to do
# SCIRL-trained policy manage because they have trained there with the reward found by SCIRL
# (data for training with LSPI cover all quadrants)
# What is cool is that we find a "good" reward in a quadrant we never saw (that was not possible before)
# But to go from this reward to a control policy, you still need data there



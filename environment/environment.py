
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
from qutip import *
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import math


tf.compat.v1.enable_v2_behavior()

"""
action_space = ControlSpace([.4, 10, 10], [.1, 1, 1])
environment = State(0.25,
                    sigmax(),
                    basis(2, 0),
                    [sigmaz(),   sigmax(),   sigmay(),   sigmam()])

control_agent = Agent(0.9, environment, action_space)
"""


class SpinQubitEnv(py_environment.PyEnvironment):

    """
    Divide the total time frame of operation into n segments of equal duration. 
    There are two possibilities of going about this. 
       1. Use a heuristic to find the time taken for the whole operation. Assume that 
          this time is our worst case time. ie our algorithm will defnitely improve the 
           time. So divide the time required by heuristic into n segments

       2. Take the interval_width as a tunable parameter. Start with a random value. And 
           find the number of time steps such that, after n steps the state obtained will have 
           the maximum fidelity

        For computational reasons we go by method 2 [it goes better with RL approach]
    TODO [SCALE]: CUrrently the implemention is limited to single qubits. Make it generic using numpy array for n-qubits

    # Set the threshold to start fine tuning [with respect to fidelity] 
    pick the user defined intial control (default 0 0 0). Based on the intial control we find the k controls from knn or kfn for the fist time segment

    Create control list which can be appended after each segment duration. 

    detuning, Omega_x and Omega_y are the control parameters (or the actions) which can be varied
    for each time segment. Based on these controls the system will evolve into a new state. We start
    with controlling only the amplitude  and not the wave type (ie its always square wave)
    TODO [SCALE] : Have a functionality for changing the wave type

    TODO [STANDARDIZE] : Ensure all the values are float for consistancy. numpy mgrid stores the value aS 0. while a float is 0.0
    # This may create inconsitancy while converting to string. This is a minor issue as the coordinates will be always 
    # used from the coordinate matrix. (Since we are restricting any other control coordinate
    """

    def __init__(self, interval_width,
                 target_unitary,
                 initial_state,
                 rabi_freq,
                 relative_detuning,
                 gamma=[0, 0],
                 noise=None):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(7,), dtype=np.float32, minimum=-1, name='observation')
        self._state = basis(2, 0)
        self._episode_ended = False
        self.fidelity = 0
        self.time_stamp = 0
        self.interval_width = interval_width
        self.max_stamp = 10*5  # 10 -> relative = 5 for omega = 10
        self.max_fidelity = 0
        self.target_unitary = target_unitary
        self.initial_state = initial_state
        self.rabi_freq = rabi_freq
        self.relative_detuning = relative_detuning
        self.gamma = gamma
        self.target_state = target_unitary*initial_state
        self.noise = noise  # noise is of the form [operator, fn]

        """
        Example for defining noise
        
        def noise(self, t, args):
            # TODO [ALGORITHM]:  make this staochastic using random variable for each segment
            h = 1
            return np.sin(10*t)*10*h/2
        noise_op = sigmaz()
        H_noise = [noise_op, noise]
        
        """

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = basis(2, 0)
        self._episode_ended = False
        self.fidelity = 0
        self.time_stamp = 0
        self.max_fidelity = 0
        #self.gamma = [random.uniform(0, 1), random.uniform(0, 1)]
        return ts.restart(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32))

    def render(self):

        return (self._state, self.fidelity)

    def set_gamma(self, gamma):
        self.gamma = gamma

    def _step(self, action):
        self.time_stamp += 1
        if self.time_stamp >= self.max_stamp:
            self._episode_ended = True
        else:
            self._episode_ended = False

        if self._episode_ended:
            self.max_fidelity = 0
            return ts.termination(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32), 0)

        H, c_ops = self.get_hamiltonian(action)

        # newly added

        c_ops = []
        # print(self.gamma)
        if self.gamma[0] > 0.0:
            H += np.sqrt(self.gamma[0]) * sigmam()

        if self.gamma[1] > 0.0:
            H += np.sqrt(self.gamma[1]) * sigmaz()

        ######

        t = np.arange(0, self.get_interval_width(), .01)
        transition_state = mesolve(H, self._state, t, c_ops=c_ops)
        self._state = transition_state.states[-1]
        new_fidelity = self.get_transition_fidelity(self._state)

        #reward = 2*new_fidelity - self.fidelity - self.max_fidelity
        #reward = reward if reward > 0 else 0
        reward = new_fidelity - self.fidelity

        self.fidelity = new_fidelity
        self.max_fidelity = new_fidelity if new_fidelity > self.max_fidelity else self.max_fidelity

        observation = [self.fidelity,
                       expect(sigmax(), self._state),
                       expect(sigmay(), self._state),
                       expect(sigmaz(), self._state),
                       reward,
                       action[0],
                       action[1]
                       ]

        if (self.fidelity > 0.9995 or self.time_stamp >= self.max_stamp):
            self._episode_ended = True
            self.max_fidelity = 0
            self.fidelity = 0
            #self.gamma = [random.uniform(0, 1), random.uniform(0, 1)]
            return ts.termination(np.array(observation, dtype=np.float32), reward=reward)

        else:
            return ts.transition(
                np.array(observation, dtype=np.float32), reward=reward, discount=0.9)

    def get_target_state(self):
        return self.target_state

    def get_current_state(self):
        return self.current_state

    def get_interval_width(self):
        return self.interval_width

    def get_fidelity(self):
        return (self.get_target_state().dag() * self.get_current_state()).norm()

    def get_transition_fidelity(self, transition_state):
        return (self.get_target_state().dag()*transition_state).norm()

    def get_hamiltonian_operator_z(self):
        return sigmaz()

    def get_hamiltonian_operator_x(self):
        return sigmax()

    def get_hamiltonian_operator_y(self):
        return sigmay()

    def get_hamiltonian_operator_noise(self):
        return self.noise

    def noise(self, t, args):
        # TODO [ALGORITHM]:  make this staochastic using random variable for each segment
        h = 1
        return np.sin(10*t)*10*h/2

    def update_state(self, next_state):
        self.current_state = next_state

    def get_hamiltonian_control(self, controls):
        h = 1
        detuning = 2*np.pi*self.relative_detuning*self.rabi_freq
        omegax = 2*np.pi*self.rabi_freq*controls[0]
        omegay = 2*np.pi*self.rabi_freq*controls[1]
        H_z = self.get_hamiltonian_operator_z()
        H_x = self.get_hamiltonian_operator_x()
        H_y = self.get_hamiltonian_operator_y()
        H_control = (detuning*H_z
                     + omegax*H_x
                     + omegay*H_y)
        return H_control

    def get_hamiltonian(self, controls):
        H_control = self.get_hamiltonian_control(controls)
        c_ops = []

        if self.gamma[0] > 0.0:
            c_ops.append(np.sqrt(self.gamma[0]) * sigmam())

        if self.gamma[1] > 0.0:
            c_ops.append(np.sqrt(self.gamma[1]) * sigmaz())

        #H_noise = [self.get_hamiltonian_operator_noise(), self.noise]
        return [H_control, c_ops]


def validate_evironment():
    validate_env = SpinQubitEnv(0.1,
                                sigmax(),
                                basis(2, 0),
                                1,
                                .1)
    utils.validate_py_environment(validate_env, episodes=5)


def get_tf_environment(x, y, interval_width, target_unitary, rabi_freq, relative_detuning):

    initial_state = math.cos(x/2)*basis(2, 0) + math.sin(x/2)*basis(2, 1)*(math.cos(y) + 1j*math.sin(y))
    initial_state = initial_state.unit()

    py_env = SpinQubitEnv(interval_width,
                          target_unitary,
                          initial_state,
                          rabi_freq,
                          relative_detuning)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    return tf_env


# %%


# %%


# %%

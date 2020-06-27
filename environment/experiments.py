# Copyright [2019] [Kevin Abraham]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tf_agents.policies.policy_saver import PolicySaver

import matplotlib as plt
from qutip import *
from agent_tf import *
from environment import *

import abc
import tensorflow as tf
import numpy as np
import tf_agents
import math
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

"""
Sample run with intial state Qobj = [[1]  and target = [[0]
                                    [0]]                [1]]

The measure of accuracy is fidelity. 
looking for a target accuracy of .99965 
The slighest of variations matters here given the growth of error 
with increase in qubits
"""


# action_space = ControlSpace([.4,10,10], [.1, 1, 1])
# environment = State(0.25,
#                     q.sigmax(),
#                     q.basis(2,0),
#                     [q.sigmaz(), q.sigmax(), q.sigmay(), q.sigmam()])

# control_agent = Agent(0.9, environment, action_space)
# control_agent.learn_rollout()


# Learning Parameters
num_iterations = 1000  # @param {type:"integer"}
collect_episodes_per_iteration = 250  # @param {type:"integer"}
# for ddpg max repaly = 2
replay_buffer_capacity = 2000  # @param {type:"integer"}

fc_layer_params = (100, 2)

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 25  # @param {type:"integer"}
num_eval_episodes = 2  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}


# Quantum Environment Parameters
interval_width = 0.1
target_unitary = sigmax()
initial_state = basis(2, 0)
rabi_freq = 1
relative_detuning = .1


def train(dummy_env, tf_agent):

    #parallel_env = ParallelPyEnvironment([SpinQubitEnv(en_configs[i]) for i in range(6)])
    #train_env = tf_py_environment.TFPyEnvironment(parallel_env)
    #gammavals = np.linspace(.01,.99, 5000)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=dummy_env.batch_size,
        max_length=replay_buffer_capacity)

    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    avg_return = compute_avg_return(dummy_env, tf_agent.policy, 10)
    returns = [avg_return]

    for x in np.linspace(0, np.pi, 10):
        for y in np.linspace(0, 2*np.pi, 10):
            for target_unitary in [sigmax()]:
                for relative_detuning in [.1, 10]:
                    for _ in range(250):
                        # for _ in range(num_iterations):
                        # num_iterations
                        # Collect a few episodes using collect_policy and save to the replay buffer.
                        #train_env,eval_env = get_environments(x, y)

                        train_env = get_tf_environment(x, y, 0.1, target_unitary, 1, relative_detuning)
                        collect_episode(
                            train_env, replay_buffer, tf_agent.collect_policy, collect_episodes_per_iteration)

                        # Use data from the buffer and update the agent's network.
                        experience = replay_buffer.gather_all()
                        train_loss = tf_agent.train(experience)
                        replay_buffer.clear()

                        step = tf_agent.train_step_counter.numpy()

                        if step % 100 == 0:
                            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

                        if True or step % eval_interval == 0:

                            #eval_py_env = SpinQubitEnv(environment_configs)
                            ##eval_gamma = [random.uniform(0,0.5),random.uniform(0,0.5)]
                            # eval_py_env.set_gamma(eval_gamma)
                            #eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
                            eval_env = get_tf_environment(x, y, 0.1, target_unitary, 1, relative_detuning)
                            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
                            print('step = {0}: Average Return = {1}'.format(step, avg_return))
                            returns.append(avg_return)
    return (step, train_loss, returns, fidelity)


def plot_training_return():
    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim(top=2)


def evaluate(tf_agent, eval_py_env):
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    data = []
    num_episodes = 1
    fidelity = []
    actions = []
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        state, fidel = eval_py_env.render()
        data.append(state)
        fidelity.append(fidel)
        while not time_step.is_last():
            action_step = tf_agent.policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            actions.append(action_step.action)
            state, fidel = eval_py_env.render()
            data.append(state)
            fidelity.append(fidel)
    return data, fidelity, actions


def evaluate_policy(policy, eval_py_env):
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    data = []
    num_episodes = 1
    fidelity = []
    actions = []
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        state, fidel = eval_py_env.render()
        data.append(state)
        fidelity.append(fidel)
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            actions.append(action_step.action)
            state, fidel = eval_py_env.render()
            data.append(state)
            fidelity.append(fidel)
    return data, fidelity, actions


def save_policy(tf_agent, fname='/content/reinforce_cz'):
    my_policy = tf_agent.collect_policy
    # save policy
    PolicySaver(my_policy).save(fname)


def create_anim(fname, data):
    sphere = Bloch()
    for i in range(len(data)):
        sphere.clear()
        sphere.add_states(data[:i])
        sphere.make_sphere()
        sphere.save(fname + str(i))


def save_controls(control_name, fname, action):
    steps = [i*environment_configs.get_interval_width() for i in range(action.shape[0])]
    plt.clf()
    plt.step(steps, action)
    plt.ylabel(control_name)
    plt.xlabel('t')
    plt.savefig(fname+'_'+control_name)


def save_fidelity(fname, fidelity):
    steps = [i*environment_configs.get_interval_width() for i in range(len(fidelity))]
    plt.clf()
    plt.plot(steps, fidelity)
    plt.ylabel('fidelity')
    plt.xlabel('step')
    plt.savefig(fname + 'fidelity'+'_')


def save_all_controls(fname, actions):
    actions = np.array(actions).reshape(len(actions), 2).transpose()
    save_controls('Omega_x', fname, actions[0])
    save_controls('Omega_y', fname, actions[1])


def save_evals(fname, result):
    save_all_controls(fname, result[2])
    save_fidelity(fname, result[1])
    #create_anim(fname, result[0] )


# %%
# for ddpg max repaly = 2
replay_buffer_capacity = 2000
dummy_env = SpinQubitEnv(interval_width=0.1,
                         target_unitary=sigmax(),
                         initial_state=basis(2, 0),
                         rabi_freq=1,
                         relative_detuning=0.1)
tf_dumm = tf_py_environment.TFPyEnvironment(dummy_env)


# %%

agent_reinforce = get_agent(dummy_env, 'reinforce', "without_noise_trained")
train_results = train(tf_dumm, agent_reinforce)
#PolicySaver(agent_reinforce.collect_policy).save('/content/drive/My Drive/qControl/policy4')

# %%

env_noisy = SpinQubitEnv(noisy_configs)
env_noisy.set_gamma([.1, .3])
eval_noisy = evaluate(agent_reinforce, env_noisy)
save_evals('/content/drive/My Drive/qControl/test2', eval_noisy)
eval_noisy[1]

# %%

initial_states = []
for x in np.linspace(0, np.pi, 10):
    for y in np.linspace(0, 2*np.pi, 10):
        initial_state = math.cos(x/2)*basis(2, 0) + math.sin(x/2)*basis(2, 1)*(math.cos(y) + 1j*math.sin(y))
        initial_state = initial_state.unit()
        initial_states.append(initial_state)
sphere = Bloch3d()
sphere.add_states(initial_states)
sphere.show()

# %%

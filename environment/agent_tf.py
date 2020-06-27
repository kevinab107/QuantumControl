import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from itertools import combinations
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.networks import actor_distribution_network
from tf_agents.trajectories import trajectory
import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
import numpy

# TODO: Encapsulate agents into a class. for ddpg max repaly = 2. ignore the input value
# of replay for ddgp
num_iterations = 1000  # @param {type:"integer"}
collect_episodes_per_iteration = 100  # @param {type:"integer"}
# for ddpg max repaly = 2
replay_buffer_capacity = 2000  # @param {type:"integer"}

fc_layer_params = (100, 2)

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 25  # @param {type:"integer"}
num_eval_episodes = 2  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}


def get_ddpg_agent(spin_py_enviroment, name):
    train_env = tf_py_environment.TFPyEnvironment(spin_py_enviroment)
    actor_network = tf_agents.agents.ddpg.actor_network.ActorNetwork(
        train_env.observation_spec(),
        train_env.action_spec())

    critic_network = tf_agents.agents.ddpg.critic_network.CriticNetwork(
        (train_env.observation_spec(),
         train_env.action_spec()), name='CriticNetwork'
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step_counter = tf.compat.v2.Variable(0)

    name = name + '_ddpg'
    tf_agent = tf_agents.agents.DdpgAgent(
        train_env.time_step_spec(), train_env.action_spec(), actor_network, critic_network,
        actor_optimizer=optimizer, critic_optimizer=optimizer, ou_stddev=1.0, ou_damping=1.0,
        target_actor_network=None, target_critic_network=None, target_update_tau=1.0,
        target_update_period=1, dqda_clipping=None, td_errors_loss_fn=None, gamma=1.0,
        reward_scale_factor=1.0, gradient_clipping=None, debug_summaries=False,
        summarize_grads_and_vars=False, train_step_counter=None, name=name
    )

    return tf_agent


def get_reinforce_agent(spin_py_environment, name):

    train_env = tf_py_environment.TFPyEnvironment(spin_py_environment)
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec())

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)
    name = name + '_reinforce'
    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter,
        name=name)

    tf_agent.initialize()

    return tf_agent


def get_agent(env, agent_type, name):
    return get_reinforce_agent(env, name) if agent_type == 'reinforce' else get_ddpg_agent(env, name)


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            print(_, time_step.observation[0])
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_episode(environment, replay_buffer, policy, num_episodes):

    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

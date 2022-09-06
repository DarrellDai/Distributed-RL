from unittest import mock

import numpy as np
import pytest
from gym import spaces
from mlagents_envs.base_env import (ActionType, BehaviorMapping, BehaviorSpec,
                                    DecisionSteps, TerminalSteps)
from unity_wrappers.envs import MultiUnityWrapper, UnityToGymWrapper


def test_gym_wrapper():
    mock_env = mock.MagicMock()
    mock_spec = create_mock_group_spec()
    mock_decision_step, mock_terminal_step = create_mock_vector_steps(
        mock_spec)
    setup_mock_unityenvironment(
        mock_env, mock_spec, mock_decision_step, mock_terminal_step
    )

    env = UnityToGymWrapper(mock_env)
    assert isinstance(env, UnityToGymWrapper)
    assert isinstance(env.reset(), np.ndarray)
    actions = env.action_space.sample()
    assert actions.shape[0] == 2
    obs, rew, done, info = env.step(actions)
    assert env.observation_space.contains(obs)
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


def test_branched_flatten():
    mock_env = mock.MagicMock()
    mock_spec = create_mock_group_spec(
        vector_action_space_type="discrete", vector_action_space_size=[2, 2, 3]
    )
    mock_decision_step, mock_terminal_step = create_mock_vector_steps(
        mock_spec, num_agents=1
    )
    setup_mock_unityenvironment(
        mock_env, mock_spec, mock_decision_step, mock_terminal_step
    )

    env = UnityToGymWrapper(mock_env, flatten_branched=True)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 12
    assert env._flattener.lookup_action(0) == [0, 0, 0]
    assert env._flattener.lookup_action(11) == [1, 1, 2]

    # Check that False produces a MultiDiscrete
    env = UnityToGymWrapper(mock_env, flatten_branched=False)
    assert isinstance(env.action_space, spaces.MultiDiscrete)


@pytest.mark.parametrize("use_uint8", [True, False], ids=["float", "uint8"])
def test_gym_wrapper_visual(use_uint8):
    mock_env = mock.MagicMock()
    mock_spec = create_mock_group_spec(
        number_visual_observations=1, vector_observation_space_size=0)
    mock_decision_step, mock_terminal_step = create_mock_vector_steps(
        mock_spec, number_visual_observations=1)
    setup_mock_unityenvironment(
        mock_env, mock_spec, mock_decision_step, mock_terminal_step
    )

    env = UnityToGymWrapper(mock_env, uint8_visual=use_uint8)
    assert isinstance(env, UnityToGymWrapper)
    assert isinstance(env.reset(), np.ndarray)
    actions = env.action_space.sample()
    assert actions.shape[0] == 2
    obs, rew, done, info = env.step(actions)
    assert env.observation_space.contains(obs)
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)

# Tests for mutliagent wrapper


def test_multi_wrapper():
    mock_env = mock.MagicMock()
    mock_spec = create_mock_group_spec()
    mock_decision_step, mock_terminal_step = create_mock_vector_steps(
        mock_spec, num_agents=3)
    setup_mock_unityenvironment(
        mock_env, mock_spec, mock_decision_step, mock_terminal_step
    )

    env = MultiUnityWrapper(mock_env)
    assert isinstance(env, MultiUnityWrapper)
    obs = env.reset()
    agent_id_0 = list(obs.keys())[0]
    assert isinstance(obs, dict)
    assert isinstance(obs[agent_id_0], np.ndarray)
    actions = {agent_id: env.action_space[agent_id].sample(
    ) for agent_id in obs.keys()}
    assert actions[agent_id_0].shape[0] == 2
    observations, rewards, dones, info = env.step(actions)
    assert env.observation_space[0].contains(obs[agent_id_0])

    obs = observations[agent_id_0]
    rew = rewards[agent_id_0]
    done = dones[agent_id_0]

    assert isinstance(observations, dict)
    assert isinstance(rewards, dict)
    assert isinstance(dones, dict)
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


def test_multi_branched_flatten():
    mock_env = mock.MagicMock()
    mock_spec = create_mock_group_spec(
        vector_action_space_type="discrete", vector_action_space_size=[2, 2, 3]
    )
    mock_decision_step, mock_terminal_step = create_mock_vector_steps(
        mock_spec, num_agents=3
    )
    setup_mock_unityenvironment(
        mock_env, mock_spec, mock_decision_step, mock_terminal_step
    )

    env = MultiUnityWrapper(mock_env, flatten_branched=True)
    assert isinstance(env.action_space[0], spaces.Discrete)
    assert env.action_space[0].n == 12
    assert env._flattener.lookup_action(0) == [0, 0, 0]
    assert env._flattener.lookup_action(11) == [1, 1, 2]

    # Check that False produces a MultiDiscrete
    env = MultiUnityWrapper(mock_env, flatten_branched=False)
    assert isinstance(env.action_space[0], spaces.MultiDiscrete)


@pytest.mark.parametrize("use_uint8", [True, False], ids=["float", "uint8"])
def test_multi_wrapper_visual(use_uint8):
    mock_env = mock.MagicMock()
    mock_spec = create_mock_group_spec(
        number_visual_observations=1, vector_observation_space_size=0)
    mock_decision_step, mock_terminal_step = create_mock_vector_steps(
        mock_spec, number_visual_observations=1, num_agents=3
    )
    setup_mock_unityenvironment(
        mock_env, mock_spec, mock_decision_step, mock_terminal_step
    )

    env = MultiUnityWrapper(mock_env, uint8_visual=use_uint8)
    assert isinstance(env, MultiUnityWrapper)
    obs = env.reset()
    agent_id_0 = list(obs.keys())[0]
    assert isinstance(obs, dict)
    assert isinstance(obs[agent_id_0], np.ndarray)

    actions = {agent_id: env.action_space[agent_id].sample(
    ) for agent_id in obs.keys()}
    assert actions[agent_id_0].shape[0] == 2
    observations, rewards, dones, info = env.step(actions)
    assert env.observation_space[0].contains(obs[agent_id_0])

    obs = observations[agent_id_0]
    rew = rewards[agent_id_0]
    done = dones[agent_id_0]
    assert isinstance(observations, dict)
    assert isinstance(rewards, dict)
    assert isinstance(dones, dict)
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


# Helper methods


def create_mock_group_spec(
    number_visual_observations=0,
    vector_action_space_type="continuous",
    vector_observation_space_size=3,
    vector_action_space_size=None,
):
    """
    Creates a mock BrainParameters object with parameters.
    """
    # Avoid using mutable object as default param
    act_type = ActionType.DISCRETE
    if vector_action_space_type == "continuous":
        act_type = ActionType.CONTINUOUS
        if vector_action_space_size is None:
            vector_action_space_size = 2
        else:
            vector_action_space_size = vector_action_space_size[0]
    else:
        if vector_action_space_size is None:
            vector_action_space_size = (2,)
        else:
            vector_action_space_size = tuple(vector_action_space_size)
    obs_shapes = [(vector_observation_space_size,)]
    for _ in range(number_visual_observations):
        obs_shapes += [(8, 8, 3)]
    return BehaviorSpec(obs_shapes, act_type, vector_action_space_size)


def create_mock_vector_steps(specs, num_agents=1, number_visual_observations=0):
    """
    Creates a mock BatchedStepResult with vector observations. Imitates constant
    vector observations, rewards, dones, and agents.

    :BehaviorSpecs specs: The BehaviorSpecs for this mock
    :int num_agents: Number of "agents" to imitate in your BatchedStepResult values.
    """
    obs = [np.array([num_agents * [1, 2, 3]]).reshape(num_agents, 3)]
    if number_visual_observations:
        obs += [np.zeros(shape=(num_agents, 8, 8, 3),
                         dtype=np.float32)] * number_visual_observations
    rewards = np.array(num_agents * [1.0])
    agents = np.array(range(0, num_agents))
    return DecisionSteps(obs, rewards, agents, None), TerminalSteps.empty(specs)


def setup_mock_unityenvironment(mock_env, mock_spec, mock_decision, mock_termination):
    """
    Takes a mock UnityEnvironment and adds the appropriate properties, defined by the mock
    GroupSpec and BatchedStepResult.

    :Mock mock_env: A mock UnityEnvironment, usually empty.
    :Mock mock_spec: An AgentGroupSpec object that specifies the params of this environment.
    :Mock mock_decision: A DecisionSteps object that will be returned at each step and reset.
    :Mock mock_termination: A TerminationSteps object that will be returned at each step and reset.
    """
    mock_env.behavior_specs = BehaviorMapping({"MockBrain": mock_spec})
    mock_env.get_steps.return_value = (mock_decision, mock_termination)

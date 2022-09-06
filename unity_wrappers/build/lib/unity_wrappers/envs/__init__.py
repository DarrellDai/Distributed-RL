import itertools
import time
from abc import ABC
from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np
from gym import error, spaces
from mlagents_envs import logging_util
from mlagents_envs.base_env import BaseEnv, DecisionSteps, TerminalSteps


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """


class MultiUnityWrapperException(error.Error):
    """
    Any error related to the multiagent wrapper of ml-agents.
    """


logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)

GymStepResult = Tuple[np.ndarray, float, bool, Dict]
MultiStepResult = Tuple[Dict[str, np.ndarray],
                        Dict[str, float], Dict[str, bool], Dict]


class UnityToGymWrapper(gym.Env):
    """
    Provides Gym wrapper for Unity Learning Environments.
    """

    def __init__(
        self,
        unity_env: BaseEnv,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
    ):
        """
        Environment initialization
        :param unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param allow_multiple_obs: If True, return a list of np.ndarrays as observations with the first elements
            containing the visual observations and the last element containing the array of vector observations.
            If False, returns a single np.ndarray containing either only a single visual observation or the array of
            vector observations.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.visual_obs = None

        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: DecisionSteps = None
        self._flattener = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs

        # Check brain configuration
        if len(self._env.behavior_specs) != 1:
            raise UnityGymException(
                "There can only be one behavior in a UnityEnvironment "
                "if it is wrapped in a gym."
            )

        self.name = list(self._env.behavior_specs.keys())[0]
        self.group_spec = self._env.behavior_specs[self.name]

        if self._get_n_vis_obs() == 0 and self._get_vec_obs_size() == 0:
            raise UnityGymException(
                "There are no observations provided by the environment."
            )

        if not self._get_n_vis_obs() >= 1 and uint8_visual:
            logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual
        if (
            self._get_n_vis_obs() + self._get_vec_obs_size() >= 2
            and not self._allow_multiple_obs
        ):
            logger.warning(
                "The environment contains multiple observations. "
                "You must define allow_multiple_obs=True to receive them all. "
                "Otherwise, only the first visual observation (or vector observation if"
                "there are no visual observations) will be provided in the observation."
            )

        # Check for number of agents in scene.
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.name)
        self._check_agents(len(decision_steps))
        self._previous_decision_step = decision_steps

        # Set action spaces
        if self.group_spec.is_action_discrete():
            branches = self.group_spec.discrete_action_branches
            if self.group_spec.action_shape == 1:
                self._action_space = spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(branches)

        else:
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )
            high = np.array([1] * self.group_spec.action_shape)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)

        # Set observations space
        list_spaces: List[gym.Space] = []
        shapes = self._get_vis_obs_shape()
        for shape in shapes:
            if uint8_visual:
                list_spaces.append(spaces.Box(
                    0, 255, dtype=np.uint8, shape=shape))
            else:
                list_spaces.append(spaces.Box(
                    0, 1, dtype=np.float32, shape=shape))
        if self._get_vec_obs_size() > 0:
            # vector observation is last
            high = np.array([np.inf] * self._get_vec_obs_size())
            list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
        if self._allow_multiple_obs:
            self._observation_space = spaces.Tuple(list_spaces)
        else:
            # only return the first one
            self._observation_space = list_spaces[0]

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object/list): the initial observation of the
        space.
        """
        self._env.reset()
        decision_step, _ = self._env.get_steps(self.name)
        n_agents = len(decision_step)
        self._check_agents(n_agents)
        self.game_over = False

        res: GymStepResult = self._single_step(decision_step)
        return res[0]

    def step(self, action: List[Any]) -> GymStepResult:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object/list): an action provided by the environment
        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information.
        """
        if self._flattener is not None:
            # Translate action into list
            action = self._flattener.lookup_action(action)

        spec = self.group_spec
        action = np.array(action).reshape((1, spec.action_size))
        self._env.set_actions(self.name, action)

        self._env.step()
        decision_step, terminal_step = self._env.get_steps(self.name)
        self._check_agents(max(len(decision_step), len(terminal_step)))
        if len(terminal_step) != 0:
            # The agent is done
            self.game_over = True
            return self._single_step(terminal_step)
        else:
            return self._single_step(decision_step)

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        if self._allow_multiple_obs:
            visual_obs = self._get_vis_obs_list(info)
            visual_obs_list = []
            for obs in visual_obs:
                visual_obs_list.append(self._preprocess_single(obs[0]))
            default_observation = visual_obs_list
            if self._get_vec_obs_size() >= 1:
                default_observation.append(self._get_vector_obs(info)[0, :])
        else:
            if self._get_n_vis_obs() >= 1:
                visual_obs = self._get_vis_obs_list(info)
                default_observation = self._preprocess_single(visual_obs[0][0])
            else:
                default_observation = self._get_vector_obs(info)[0, :]

        if self._get_n_vis_obs() >= 1:
            visual_obs = self._get_vis_obs_list(info)
            self.visual_obs = self._preprocess_single(visual_obs[0][0])

        done = isinstance(info, TerminalSteps)

        return (default_observation, info.reward[0], done, {"step": info})

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        else:
            return single_visual_obs

    def _get_n_vis_obs(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                result += 1
        return result

    def _get_vis_obs_shape(self) -> List[Tuple]:
        result: List[Tuple] = []
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                result.append(shape)
        return result

    def _get_vis_obs_list(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> np.ndarray:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 1:
                result += shape[0]
        return result

    def render(self, mode="rgb_array"):
        return self.visual_obs

    def close(self) -> None:
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        """Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        logger.warning("Could not seed environment %s", self.name)
        return

    @staticmethod
    def _check_agents(n_agents: int) -> None:
        if n_agents > 1:
            raise UnityGymException(
                f"There can only be one Agent in the environment but {n_agents} were detected."
            )

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -float("inf"), float("inf")

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """
    # Unity class

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action)
                       for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        return self.action_lookup[action]


class MultiUnityWrapper():
    """
    Provides wrapper for Unity Learning Environments, supporting multiagents.
    """
    # Implemented class. Implements: rllib.MultiEnv.
    # (not done because rllib cannot be installed on windows for now)

    def __init__(
        self,
        unity_env: BaseEnv,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
    ):
        """
        Environment initialization
        :param unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
        :param use_visual: Whether to use visual observation or vector observation.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param allow_multiple_obs: If True, return a list of np.ndarrays as observations with the first elements
            containing the visual observations and the last element containing the array of vector observations.
            If False, returns a single np.ndarray containing either only a single visual observation or the array of
            vector observations.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.visual_obs = None

        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: DecisionSteps = None
        self._flattener = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs

        self.behaviour_names = [
            name for name in self._env.behavior_specs.keys()]

        # Check for number of agents in scene.
        self._n_agents = 0
        self._env.reset()
        self._agent_id_to_behaviour_name = {}
        self._agents_dict = {}
        for name in self.behaviour_names:
            decision_steps, _ = self._env.get_steps(name)
            self._agents_dict[name] = []
            for agent_id in decision_steps.agent_id:
                self._agent_id_to_behaviour_name[agent_id] = name
                self._agents_dict[name].append(agent_id)
                self._n_agents += 1
            self._previous_decision_step = decision_steps

        if self._get_n_vis_obs() == 0 and self._get_vec_obs_size() == 0:
            raise MultiUnityWrapperException(
                "There are no observations provided by the environment."
            )

        if not all(self._get_n_vis_obs().values()) >= 1 and uint8_visual:
            logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual

        if all(self._get_n_vis_obs().values()) + all(self._get_vec_obs_size().values()) >= 2 and not self._allow_multiple_obs:
            logger.warning(
                "The environment contains multiple observations. "
                "You must define allow_multiple_obs=True to receive them all. "
                "Otherwise, only the first visual observation (or vector observation if"
                "there are no visual observations) will be provided in the observation."
            )

        # Set observation and action spaces
        self._action_space = {}
        self._observation_space = {}
        vec_obs_size_dict = self._get_vec_obs_size()
        shape_dict = self._get_vis_obs_shape()
        for behaviour_name, group_spec in self._env.behavior_specs.items():
            # Set observations space
            if group_spec.is_action_discrete():
                branches = group_spec.discrete_action_branches
                if group_spec.action_shape == 1:
                    action_space = spaces.Discrete(branches[0])
                else:
                    if flatten_branched:
                        self._flattener = ActionFlattener(branches)
                        action_space = self._flattener.action_space
                    else:
                        action_space = spaces.MultiDiscrete(branches)

            else:
                if flatten_branched:
                    logger.warning(
                        "The environment has a non-discrete action space. It will "
                        "not be flattened."
                    )
                high = np.array([1] * group_spec.action_shape)
                action_space = spaces.Box(-high, high, dtype=np.float32)

            # Set observations space
            list_spaces: List[gym.Space] = []
            shapes = shape_dict[behaviour_name]
            for shape in shapes:
                if uint8_visual:
                    list_spaces.append(spaces.Box(
                        0, 255, dtype=np.uint8, shape=shape))
                else:
                    list_spaces.append(spaces.Box(
                        0, 1, dtype=np.float32, shape=shape))
            if vec_obs_size_dict[behaviour_name] > 0:
                # vector observation is last
                high = np.array([np.inf] * vec_obs_size_dict[behaviour_name])
                list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
            if self._allow_multiple_obs:
                observation_space = spaces.Tuple(list_spaces)
            else:
                observation_space = list_spaces[0]  # only return the first one

            # Assign spaces to agents
            for agent_id in self._agents_dict[behaviour_name]:
                self._observation_space[agent_id] = observation_space
                self._action_space[agent_id] = action_space

    def reset(self) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        """
        Resets the state of the environment and returns an initial observation.
        Returns: observation (object/list): the initial observation of the
        space.
        """
        self._env.reset()
        decision_steps_dict = {behaviour_name: self._env.get_steps(
            behaviour_name)[0] for behaviour_name in self.behaviour_names}
        n_agents = sum([len(decision_step)
                        for decision_step in decision_steps_dict.values()])
        self._check_agents(n_agents)
        self.game_over = False

        res: GymStepResult = self._single_step(decision_steps_dict)
        # Returns only observation
        return res[0]

    def step(self, action_dict: Dict) -> MultiStepResult:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action_dict (dict): dict of actions provided by all agents
        Returns:
            observation (dict): agents' observations of the current environment
            reward (dict) : amount of rewards returned after previous action
            done (dict): whether the episode has ended for each agent.
            info (dict): contains auxiliary diagnostic information.
        """
        if self._flattener is not None:
            for agent, action in action_dict.items():
                # Translate action into list
                action_dict[agent] = self._flattener.lookup_action(action)

        for agent_id, action in action_dict.items():
            behaviour_name = self._agent_id_to_behaviour_name[agent_id]
            self._env.set_action_for_agent(behaviour_name, agent_id, action)

        self._env.step()
        decision_steps_dict, terminal_steps_dict = {}, {}

        for behaviour_name in self.behaviour_names:
            decision_step, terminal_step = self._env.get_steps(behaviour_name)
            decision_steps_dict[behaviour_name] = decision_step
            terminal_steps_dict[behaviour_name] = terminal_step

        self._check_agents(
            max(len(decision_steps_dict), len(terminal_steps_dict)))

        decision_obs_dict, decision_reward_dict, decision_done_dict, decision_info = self._single_step(
            decision_steps_dict)
        if len(terminal_step) != 0:
            # At least one agent is done
            _terminal_obs_dict, terminal_reward_dict, terminal_done_dict, terminal_info = self._single_step(
                terminal_steps_dict)
        else:
            terminal_reward_dict, terminal_done_dict, terminal_info = {}, {}, {}

        # Create MultiStepResult dicts
        # Episode is done: no terminal_obs
        obs_dict = decision_obs_dict
        reward_dict = {**decision_reward_dict, **terminal_reward_dict}
        done_dict = {**decision_done_dict, **terminal_done_dict}
        info_dict = {"decision_step": decision_info,
                     "terminal_step": terminal_info}

        # Game is over when all agents are done
        done_dict["__all__"] = self.game_over = (all(done_dict.values()) and len(
            done_dict.values()) == self._n_agents)
        return (obs_dict, reward_dict, done_dict, info_dict)

    def _single_step(self, info_dict: Dict[str, Tuple[DecisionSteps, TerminalSteps]]) -> GymStepResult:
        obs_dict, reward_dict, done_dict = {}, {}, {}
        vec_obs_size = self._get_vec_obs_size()
        n_vis_obs = self._get_n_vis_obs()

        for behaviour_name, info in info_dict.items():
            default_observation = None
            if self._allow_multiple_obs:
                visual_obs = self._get_vis_obs_list(info)
                visual_obs_list = []
                for obs in visual_obs:
                    visual_obs_list.append(self._preprocess_single(obs[0]))
                default_observation = visual_obs_list
                if vec_obs_size[behaviour_name] >= 1:
                    default_observation.append(
                        self._get_vector_obs(info))
            else:
                if n_vis_obs[behaviour_name] >= 1:
                    visual_obs = self._get_vis_obs_list(info)
                    default_observation = self._preprocess_single(
                        visual_obs[0][0])
                else:
                    obs_dict.update(self._get_vector_obs(
                        info))

            if n_vis_obs[behaviour_name] >= 1:
                visual_obs = self._get_vis_obs_list(info)
                self.visual_obs = self._preprocess_single(visual_obs[0][0])

            done = isinstance(info, TerminalSteps)
            for agent_id in info.agent_id:
                # Add reward and done
                agent_index = info.agent_id_to_index[agent_id]
                reward_dict[agent_id] = info.reward[agent_index]
                done_dict[agent_id] = done
                if default_observation is not None:
                    obs_dict[agent_id] = default_observation

        return (obs_dict, reward_dict, done_dict, info)

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        else:
            return single_visual_obs

    def _get_n_vis_obs(self) -> Dict:
        n_vis_obs_dict = {}
        for behaviour_name, group_spec in self._env.behavior_specs.items():
            result = 0
            for shape in group_spec.observation_shapes:
                if len(shape) == 3:
                    result += 1
            n_vis_obs_dict[behaviour_name] = result
        return n_vis_obs_dict

    def _get_vis_obs_shape(self) -> Dict[str, List[Tuple]]:
        vis_obs_shape_dict = {}
        for behaviour_name, group_spec in self._env.behavior_specs.items():
            result: List[Tuple] = []
            for shape in group_spec.observation_shapes:
                if len(shape) == 3:
                    result.append(shape)
            vis_obs_shape_dict[behaviour_name] = result

        return vis_obs_shape_dict

    def _get_vis_obs_list(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> Dict[str, np.ndarray]:
        vector_obs_dict = {}
        for agents_obs in step_result.obs:
            if len(agents_obs.shape) == 2:
                for agent_id, obs in zip(step_result.agent_id, agents_obs):
                    vector_obs_dict[agent_id] = obs
        return vector_obs_dict

    def _get_vec_obs_size(self) -> Dict:
        vec_obs_size_dict = {}
        for behaviour_name, group_spec in self._env.behavior_specs.items():
            result = 0
            for shape in group_spec.observation_shapes:
                if len(shape) == 1:
                    result += shape[0]
            vec_obs_size_dict[behaviour_name] = result
        return vec_obs_size_dict

    def render(self, mode="rgb_array"):
        return self.visual_obs

    def close(self) -> None:
        """
        Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        """
        Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        logger.warning("Could not seed environment")
        return

    # This method is a staticmethod in UnityToGym but here we need the number of agents (self._n_agents) in the env!
    def _check_agents(self, n_agents: int) -> None:
        if n_agents > self._n_agents:
            raise MultiUnityWrapperException(
                f"There can only be {self._n_agents} Agents in the environment but {n_agents} were detected."
            )

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        """
        Range in which rewards stand

        Returns:
            Tuple[float, float]: (-inf, inf)
        """
        return -float("inf"), float("inf")

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        'List of action space corresponding to each agent'
        return self._action_space

    @property
    def observation_space(self):
        'List of observation space corresponding to each agent'
        return self._observation_space

    # Does not exist anymore in UnityToGym (one agent) but it makes sense here.
    @property
    def number_agents(self):
        'Number of agents in the env'
        return self._n_agents

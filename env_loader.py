import earl_benchmark
import gym
import numpy as np
import os
import pickle
from types import SimpleNamespace

from backend.wrappers import (
    ActionRepeatWrapper,
    ObsActionDTypeWrapper,
    ExtendedTimeStepWrapper,
    ActionScaleWrapper,
    DMEnvFromGymWrapper,
    GymEnvFromGymGoalEnvWrapper,
)

# for every environment, add an entry for the configuration of the environment
# make a default configuration for environment, the user can change the parameters by passing it to the constructor.

# number of initial states being provided to the user
# for deterministic initial state distributions, it should be 1
# for stochastic initial state distributions, sample the distribution randomly and save those samples for consistency
deployment_eval_config = {
    'fetch_reach': {
        'num_initial_state_samples': 100,
        'num_goals': 100,
        'train_horizon': int(1e5),
        'eval_horizon': 50,
    },
    'fetch_push': {
        'num_initial_state_samples': 100,
        'num_goals': 100,
        'train_horizon': int(1e5),
        'eval_horizon': 50,
    },
    'fetch_pickandplace': {
        'num_initial_state_samples': 100,
        'num_goals': 100,
        'train_horizon': int(1e5),
        'eval_horizon': 50,
    },
    'fetch_slide': {
        'num_initial_state_samples': 100,
        'num_goals': 100,
        'train_horizon': int(1e5),
        'eval_horizon': 50,
    },
    'point_umaze': {
        'num_initial_state_samples': 1,
        'num_goals': 1,
        'train_horizon': int(2e5),
        'eval_horizon': 100,
    },
    
}
deployment_eval_config['fetch_reach_ergodic'] = deployment_eval_config['fetch_reach']
deployment_eval_config['fetch_push_ergodic'] = deployment_eval_config['fetch_push']
deployment_eval_config['fetch_pickandplace_ergodic'] = deployment_eval_config['fetch_pickandplace']
deployment_eval_config['fetch_reach_ergodic2'] = deployment_eval_config['fetch_reach']
deployment_eval_config['fetch_push_ergodic2'] = deployment_eval_config['fetch_push']
deployment_eval_config['fetch_pickandplace_ergodic2'] = deployment_eval_config['fetch_pickandplace']

# for continuing evaluation, only set the training horizons and goal/task change frequency.
continuing_eval_config = {
    'fetch_reach': {
        'num_initial_state_samples': 100,
        'num_goals': 100,
        'train_horizon': int(1e5),
        'goal_change_frequency': 100,
    },
    'fetch_push': {
        'num_initial_state_samples': 100,
        'num_goals': 100,
        'train_horizon': int(1e5),
        'goal_change_frequency': 100,
    },
    'fetch_pickandplace': {
        'num_initial_state_samples': 100,
        'num_goals': 100,
        'train_horizon': int(1e5),
        'goal_change_frequency': 100,
    },
    'fetch_slide': {
        'num_initial_state_samples': 100,
        'num_goals': 100,
        'train_horizon': int(1e5),
        'goal_change_frequency': 100,
    },
    'point_umaze': {
        'num_initial_state_samples': 1,
        'num_goals': 1,
        'train_horizon': int(1e5),
        'goal_change_frequency': 200,
    },
    
}
continuing_eval_config['fetch_reach_ergodic'] = continuing_eval_config['fetch_reach']
continuing_eval_config['fetch_push_ergodic'] = continuing_eval_config['fetch_push']
continuing_eval_config['fetch_pickandplace_ergodic'] = continuing_eval_config['fetch_pickandplace']
continuing_eval_config['fetch_reach_ergodic2'] = continuing_eval_config['fetch_reach']
continuing_eval_config['fetch_push_ergodic2'] = continuing_eval_config['fetch_push']
continuing_eval_config['fetch_pickandplace_ergodic2'] = continuing_eval_config['fetch_pickandplace']

class GymEnvs(object): # gym envs wrapped in EARL format
    def __init__(self,
                # parameters that need to be set for every environment
                env_name,
                reward_type='sparse',
                setup_as_lifelong_learning=False,
                full_state_goal=False, 
                reset_train_env_at_goal=False,
                # parameters that have default values in the config
                **kwargs):
        self._env_name = env_name
        self._reward_type = reward_type
        self._setup_as_lifelong_learning = setup_as_lifelong_learning
        self._full_state_goal = full_state_goal    
        self._reset_train_env_at_goal = reset_train_env_at_goal
        self._kwargs = kwargs

        # resolve to default parameters if not provided by the user
        if not self._setup_as_lifelong_learning:
            self._train_horizon = kwargs.get('train_horizon', deployment_eval_config[env_name]['train_horizon'])
            self._eval_horizon = kwargs.get('eval_horizon', deployment_eval_config[env_name]['eval_horizon'])
            self._num_initial_state_samples = kwargs.get('num_initial_state_samples', deployment_eval_config[env_name]['num_initial_state_samples'])
            self._num_goals = kwargs.get('num_goals', deployment_eval_config[env_name]['num_goals'])

            self._train_env = self.get_train_env()
            self._eval_env = self.get_eval_env()
        else:
            self._train_horizon = kwargs.get('train_horizon', continuing_eval_config[env_name]['train_horizon'])
            self._num_initial_state_samples = kwargs.get('num_initial_state_samples', continuing_eval_config[env_name]['num_initial_state_samples'])
            self._num_goals = kwargs.get('num_goals', continuing_eval_config[env_name]['num_goals'])
            self._goal_change_frequency = kwargs.get('goal_change_frequency', continuing_eval_config[env_name]['goal_change_frequency'])
            self._train_env = self.get_train_env(lifelong=True)

    def get_train_env(self, lifelong=False):
        if self._env_name == 'fetch_reach':
            from gym_robotics.envs import FetchReachEnv
            train_env = FetchReachEnv(reward_type=self._reward_type)
        elif self._env_name == 'fetch_push':
            from gym_robotics.envs import FetchPushEnv
            train_env = FetchPushEnv(reward_type=self._reward_type)
        elif self._env_name == 'fetch_pickandplace':
            from gym_robotics.envs import FetchPickAndPlaceEnv
            train_env = FetchPickAndPlaceEnv(reward_type=self._reward_type)
        elif self._env_name == 'fetch_slide':
            from gym_robotics.envs import FetchSlideEnv
            train_env = FetchSlideEnv(reward_type=self._reward_type)
        elif self._env_name == 'fetch_reach_ergodic':
            from envs import FetchReachErgodicEnv
            train_env = FetchReachErgodicEnv(reward_type=self._reward_type, reset_at_goal=self._reset_train_env_at_goal)
        elif self._env_name == 'fetch_push_ergodic':
            from envs import FetchPushErgodicEnv
            train_env = FetchPushErgodicEnv(reward_type=self._reward_type, full_state_goal= self._full_state_goal)
        elif self._env_name == 'fetch_pickandplace_ergodic':
            from envs import FetchPickAndPlaceErgodicEnv
            train_env = FetchPickAndPlaceErgodicEnv(reward_type=self._reward_type, full_state_goal= self._full_state_goal)
        elif self._env_name == 'fetch_push_ergodic2':
            from envs import FetchPushErgodicEnv2
            train_env = FetchPushErgodicEnv2(reward_type=self._reward_type, full_state_goal= self._full_state_goal, reset_at_goal=self._reset_train_env_at_goal)
        elif self._env_name == 'fetch_pickandplace_ergodic2':
            from envs import FetchPickAndPlaceErgodicEnv2
            train_env = FetchPickAndPlaceErgodicEnv2(reward_type=self._reward_type, full_state_goal= self._full_state_goal, reset_at_goal=self._reset_train_env_at_goal)
        elif self._env_name == 'point_umaze':
            from envs import PointUMazeGoalEnv
            train_env = PointUMazeGoalEnv(reward_type=self._reward_type, reset_at_goal=self._reset_train_env_at_goal)
        
        train_env = earl_benchmark.persistent_state_wrapper.PersistentStateWrapper(train_env, episode_horizon=self._train_horizon)

        if not lifelong:
            return train_env
        else:
            return earl_benchmark.lifelong_wrapper.LifelongWrapper(train_env, self._goal_change_frequency)

    def get_eval_env(self):
        if self._env_name == 'fetch_reach':
            from gym_robotics.envs import FetchReachEnv
            eval_env = FetchReachEnv(reward_type=self._reward_type)
        elif self._env_name == 'fetch_push':
            from gym_robotics.envs import FetchPushEnv
            eval_env = FetchPushEnv(reward_type=self._reward_type)
        elif self._env_name == 'fetch_pickandplace':
            from gym_robotics.envs import FetchPickAndPlaceEnv
            eval_env = FetchPickAndPlaceEnv(reward_type=self._reward_type)
        elif self._env_name == 'fetch_slide':
            from gym_robotics.envs import FetchSlideEnv
            eval_env = FetchSlideEnv(reward_type=self._reward_type)
        elif self._env_name == 'fetch_reach_ergodic':
            from envs import FetchReachErgodicEnv
            eval_env = FetchReachErgodicEnv(reward_type=self._reward_type, reset_at_goal=self._reset_train_env_at_goal)
        elif self._env_name == 'fetch_push_ergodic':
            from envs import FetchPushErgodicEnv
            eval_env = FetchPushErgodicEnv(reward_type=self._reward_type, full_state_goal= self._full_state_goal)
        elif self._env_name == 'fetch_pickandplace_ergodic':
            from envs import FetchPickAndPlaceErgodicEnv
            eval_env = FetchPickAndPlaceErgodicEnv(reward_type=self._reward_type, full_state_goal= self._full_state_goal)
        elif self._env_name == 'fetch_push_ergodic2':
            from envs import FetchPushErgodicEnv2
            eval_env = FetchPushErgodicEnv2(reward_type=self._reward_type, full_state_goal= self._full_state_goal, reset_at_goal=self._reset_train_env_at_goal)
        elif self._env_name == 'fetch_pickandplace_ergodic2':
            from envs import FetchPickAndPlaceErgodicEnv2
            eval_env = FetchPickAndPlaceErgodicEnv2(reward_type=self._reward_type, full_state_goal= self._full_state_goal, reset_at_goal=self._reset_train_env_at_goal)
        elif self._env_name == 'point_umaze':
            from envs import PointUMazeGoalEnv
            eval_env = PointUMazeGoalEnv(reward_type=self._reward_type, reset_at_goal=self._reset_train_env_at_goal)
        
        return earl_benchmark.persistent_state_wrapper.PersistentStateWrapper(eval_env, episode_horizon=self._eval_horizon)

    def has_demos(self):
        return False # gym envs does not have any demos by default

    def get_envs(self):
        if not self._setup_as_lifelong_learning:
            return self._train_env, self._eval_env
        else:
            return self._train_env

    def get_initial_states(self, num_samples=None):
        '''
        Always returns initial state of the shape N x state_dim
        '''
        if num_samples is None:
            num_samples = self._num_initial_state_samples
        # make a new copy of environment to ensure that related parameters do not get affected by collection of reset states
        cur_env = self.get_eval_env()
        reset_states = [cur_env.reset()['achieved_goal'] for _ in range(num_samples)]
        return np.stack(reset_states)

    def get_goal_states(self, num_samples=None):
        if num_samples is None:
            num_samples = self._num_goals
        if self._env_name in ['fetch_reach', 'fetch_push', 'fetch_pickandplace', 'fetch_slide', \
                              'fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', \
                              'fetch_push_ergodic2', 'fetch_pickandplace_ergodic2', \
                              'point_umaze']:
            cur_env = self.get_eval_env()
            goal_states = [cur_env._sample_goal() for _ in range(num_samples)]
            return np.stack(goal_states)

    def get_env_specs(self):
        if self._env_name in ['fetch_reach', 'fetch_push', 'fetch_pickandplace', 'fetch_slide', \
                              'fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', \
                              'fetch_push_ergodic2', 'fetch_pickandplace_ergodic2', \
                              'point_umaze']:
            cur_env = self.get_eval_env()
            obs = cur_env.reset()
            return SimpleNamespace(state_dim=obs['observation'].shape[0], goal_dim=obs['achieved_goal'].shape[0])

    def get_demonstrations(self):
        # use the current file to locate the demonstrations
        base_path = os.path.abspath(__file__)
        demo_dir = os.path.join(os.path.dirname(base_path), 'demonstrations')
        try:
            forward_demos = pickle.load(open(os.path.join(demo_dir, self._env_name, 'forward/demo_data.pkl'), 'rb'))
            reverse_demos = pickle.load(open(os.path.join(demo_dir, self._env_name, 'reverse/demo_data.pkl'), 'rb'))
            return forward_demos, reverse_demos
        except:
            print('please download the demonstrations corresponding to ', self._env_name)

def make(*args, **kwargs):
    try:
        return make_earl(*args, **kwargs)
    except:
        return make_gym(*args, **kwargs)

def make_earl(name, frame_stack, action_repeat, reset_train_env_at_goal=False):
    env_loader = earl_benchmark.EARLEnvs(
        name,
        reward_type="sparse",
        reset_train_env_at_goal=reset_train_env_at_goal,
    )
    train_env, eval_env = env_loader.get_envs()
    reset_goals = env_loader.get_initial_states() # returns goal space \phi(s) if available otherwise returns state space s
    goals = env_loader.get_goal_states() # returns goal space \phi(s) if available otherwise returns state space s
    if env_loader.has_demos():
        forward_demos, backward_demos = env_loader.get_demonstrations()
    else:
        forward_demos, backward_demos = None, None

    # add wrappers
    train_env = DMEnvFromGymWrapper(train_env)
    train_env = ObsActionDTypeWrapper(train_env, np.float32, np.float32)
    train_env = ActionRepeatWrapper(train_env, action_repeat)
    train_env = ActionScaleWrapper(train_env, minimum=-1.0, maximum=+1.0)
    train_env = ExtendedTimeStepWrapper(train_env) 

    eval_env = DMEnvFromGymWrapper(eval_env)
    eval_env = ObsActionDTypeWrapper(eval_env, np.float32, np.float32)
    eval_env = ActionRepeatWrapper(eval_env, action_repeat)
    eval_env = ActionScaleWrapper(eval_env, minimum=-1.0, maximum=+1.0)
    eval_env = ExtendedTimeStepWrapper(eval_env)

    env_specs = SimpleNamespace(
        state_dim=reset_goals.shape[-1],
        goal_dim=goals.shape[-1],
        obs_dim=train_env.observation_spec().shape[0],
    )

    return train_env, eval_env, reset_goals, goals, forward_demos, backward_demos, env_specs

def make_gym(name, frame_stack, action_repeat, reset_train_env_at_goal=False):
    env_loader = GymEnvs(
        name,
        reward_type="sparse",
        reset_train_env_at_goal=False,
    )
    train_env, eval_env = env_loader.get_envs()
    reset_goals = env_loader.get_initial_states() # returns goal space \phi(s) if available otherwise returns state space s
    goals = env_loader.get_goal_states() # returns goal space \phi(s) if available otherwise returns state space s
    env_specs = env_loader.get_env_specs()
    if env_loader.has_demos():
        forward_demos, backward_demos = env_loader.get_demonstrations()
    else:
        forward_demos, backward_demos = None, None

    # add wrappers
    train_env = GymEnvFromGymGoalEnvWrapper(train_env)
    train_env = DMEnvFromGymWrapper(train_env)
    train_env = ObsActionDTypeWrapper(train_env, np.float32, np.float32)
    train_env = ActionRepeatWrapper(train_env, action_repeat)
    train_env = ActionScaleWrapper(train_env, minimum=-1.0, maximum=+1.0)
    train_env = ExtendedTimeStepWrapper(train_env) 

    eval_env = GymEnvFromGymGoalEnvWrapper(eval_env)
    eval_env = DMEnvFromGymWrapper(eval_env)
    eval_env = ObsActionDTypeWrapper(eval_env, np.float32, np.float32)
    eval_env = ActionRepeatWrapper(eval_env, action_repeat)
    eval_env = ActionScaleWrapper(eval_env, minimum=-1.0, maximum=+1.0)
    eval_env = ExtendedTimeStepWrapper(eval_env)

    env_specs.obs_dim = train_env.observation_spec().shape[0]

    return train_env, eval_env, reset_goals, goals, forward_demos, backward_demos, env_specs

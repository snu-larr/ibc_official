import numpy as np
import gym            
from gym.spaces import Box
import torch

class NonEpisodicWrapper(object):
    def __init__(self, env, env_name, forward_env_obs_type='state_goal', backward_env_obs_type='state_goal'):
        self.env = env
        self.env_name = env_name
        # 'state_goal' (concatenated), 'state', 'goal_dict'
        self.forward_env_obs_type = forward_env_obs_type
        self.backward_env_obs_type = backward_env_obs_type
        self.option = 'forward'

        if 'tabletop' in env_name:
            obs_dim = 6
            goal_dim = 6
        elif env_name=='sawyer_door':
            if self.env.add_velocity_info=='door':
                obs_dim = 10
            elif self.env.add_velocity_info=='ee_door':
                obs_dim = 13
            else:
                obs_dim = 7
            goal_dim = 7
        elif env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
            obs_dim = 25
            if self.env.full_state_goal:
                goal_dim = 6
            else:
                goal_dim = 3
        elif env_name in ['fetch_reach_ergodic']:
            obs_dim = 10
            goal_dim = 3
        elif env_name in ['point_umaze']:
            obs_dim = 7
            goal_dim = 2
        else:
            raise NotImplementedError
            
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim

        lows = self.env.observation_space.low[:obs_dim]
        highs = self.env.observation_space.high[:obs_dim]

        self.state_space = gym.spaces.Box(lows, highs, dtype=np.float32)
        self.state_goal_space = self.env.observation_space

        self.forward_observation_space = self._get_observation_space('forward')
        self.backward_observation_space = self._get_observation_space('backward')

    def set_option(self, option, *args, **kwargs):
        self.option = option
        assert self.option =='forward' or self.option=='backward', "option should be either forward or backward"
    
    def convert_obs(self, obs, option, obs_info= None):
        if option=='forward':
            # from backward obs to forward obs
            if self.backward_env_obs_type=='state' and self.forward_env_obs_type =='state_goal':
                goal = obs_info['goal']
                obs = np.concatenate([obs, goal], axis =-1)
            elif self.backward_env_obs_type=='state_goal' and self.forward_env_obs_type =='state_goal':
                pass
            elif self.backward_env_obs_type=='state' and self.forward_env_obs_type =='state':
                pass
            else:
                raise NotImplementedError
        
        elif option=='backward':
            # from forward obs to backward obs
            if self.forward_env_obs_type=='state_goal' and self.backward_env_obs_type =='state':
                obs = self._get_obs_by_obs_type(obs, option)
            elif self.forward_env_obs_type=='state_goal' and self.backward_env_obs_type =='state_goal':
                pass
            elif self.backward_env_obs_type=='state' and self.forward_env_obs_type =='state':                
                pass
            else:
                raise NotImplementedError
        else :
            raise NotImplementedError

        return obs

    def _get_observation_space(self, option):
        if option=='forward':
            if self.forward_env_obs_type=='state_goal':            
                observation_space = self.state_goal_space
            elif self.forward_env_obs_type=='state':
                observation_space = self.state_space
            else:
                raise NotImplementedError
        
        elif option=='backward':
            if self.backward_env_obs_type=='state_goal':            
                observation_space = self.state_goal_space
            elif self.backward_env_obs_type=='state':
                observation_space = self.state_space
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return observation_space

    def _get_obs_by_obs_type(self, obs, option = None):
        '''
        Assume env returns obs, goal concatenation
        '''
        if option=='forward':
            if self.forward_env_obs_type=='state_goal':            
                pass
            elif self.forward_env_obs_type=='state':
                if 'tabletop' in self.env_name:
                    obs = obs[..., :6]
                elif 'sawyer' in self.env_name:
                    assert self.env.add_velocity_info is None
                    obs = obs[..., :7]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
        elif option=='backward':
            if self.backward_env_obs_type=='state_goal':            
                pass
            elif self.backward_env_obs_type=='state':
                if 'tabletop' in self.env_name:
                    obs = obs[..., :6]
                elif 'sawyer' in self.env_name:
                    assert self.env.add_velocity_info is None
                    obs = obs[..., :7]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return obs
    

    def get_pure_obs(self, obs):
        if 'tabletop' in self.env_name:            
            return obs[..., :6]
        elif 'sawyer' in self.env_name:
            if self.env_name=='sawyer_door':
                if self.env.add_velocity_info=='door':
                    return obs[..., :10]
                elif self.env.add_velocity_info=='ee_door':
                    return obs[..., :13]
                else:
                    return obs[..., :7]
            else:
                return obs[..., :7]
        else:
            raise NotImplementedError
    
    def replace_goal_in_obs(self, obs, goal):
        if 'tabletop' in self.env_name:            
            obs[..., 6:] = goal.copy()            
        elif 'sawyer' in self.env_name:
            if self.env_name=='sawyer_door':
                if self.env.add_velocity_info=='door':
                    obs[..., 10:] = goal.copy()
                elif self.env.add_velocity_info=='ee_door':
                    obs[..., 13:] = goal.copy()
                else:
                    obs[..., 7:] = goal.copy()
            else:
                obs[..., 7:] = goal.copy()
        elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:        
            if self.env.full_state_goal:
                assert obs.shape[-1]==37
                obs[..., -6:] = goal.copy()
            else:
                assert obs.shape[-1]==31
                obs[..., -3:] = goal.copy()
        elif self.env_name in ['fetch_reach_ergodic']:        
            assert obs.shape[-1]==16
            obs[..., -3:] = goal.copy()
        elif self.env_name in ['point_umaze']:        
            assert obs.shape[-1]==11
            obs[..., -2:] = goal.copy()
        else:
            raise NotImplementedError        
        return obs
    
    def get_object_states_only_from_goal(self, goal):
        if self.env_name in ['sawyer_door']:
            return goal[..., 4:7]

        elif self.env_name == 'tabletop_manipulation':
            raise NotImplementedError
        
        else:
            raise NotImplementedError
    
    def get_gripper_states_only_from_goal(self, goal):
        if self.env_name in ['sawyer_door']:
            return goal[..., :3]

        elif self.env_name == 'tabletop_manipulation':
            raise NotImplementedError
        
        else:
            raise NotImplementedError

    def get_pure_goal_from_obs(self, obs):
        # Assume obs : [state, ag, dg] format
        if self.env_name in ['sawyer_door']:
            if self.env.add_velocity_info=='door':
                return obs[..., 14:17]
            elif self.env.add_velocity_info=='ee_door':
                return obs[..., 17:20]
            else:
                return obs[..., 11:14]
        elif self.env_name == 'tabletop_manipulation':
            return obs[..., 6:-2]

        elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
            if self.env.full_state_goal:
                assert obs.shape[-1]==37
                return obs[..., -6:]
            else:
                assert obs.shape[-1]==31
                return obs[..., -3:]

        elif self.env_name in ['fetch_reach_ergodic']:
            assert obs.shape[-1]==16
            return obs[..., -3:]

        elif self.env_name in ['point_umaze']:
            assert obs.shape[-1]==11
            return obs[..., -2:]
        else:
            raise NotImplementedError
            
        
    def convert_state_goal_to_state(self, obs):
        return self.get_pure_obs(obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)        
        return self._get_obs_by_obs_type(next_obs, self.option).astype(np.float32), reward, done, info
    
    def reset(self):
        return self._get_obs_by_obs_type(self.env.reset(), self.option).astype(np.float32)


    def __getattr__(self, attrname):
        return getattr(self.env, attrname)


class StateWrapper(object):
    def __init__(self, env) -> None:
        self.env = env
        
    def action_spec(self):
        return self.env.action_space
    
    def observation_spec(self, option=None):
        if option is None:
            return self.env.observation_space
        elif option=='forward':
            return self.env.forward_observation_space
        elif option=='backward':
            return self.env.backward_observation_space
    
    # @property
    # def observation_space(self):
    #     return self.observation_spec()

    
    def __getattr__(self, attrname):
        return getattr(self.env, attrname)



def concatenate(*args):
    arg_list = [arg for arg in args]    
    if torch.is_tensor(arg_list[0]):
        return torch.cat(arg_list, dim =-1)
    else:
        return np.concatenate(arg_list, axis=-1)



# For EARL envs (already unwrapped env)
class WraptoGoalEnv(object): 
    '''
    NOTE : Make the env as a goal env
    '''
    
    def __init__(self, env, env_name = None, convert_goal_to_reach_object=False, sparse_reward_type='negative'):
        
        self.env = env        
        self.env_name = env_name
        # self.action_space = self.env.action_space
        # self.spaces = list(self.env.observation_space.spaces.values())
        # obs = self.env._get_obs()
        
        self.reduced_key_order = ['observation', 'desired_goal'] # assume observation==achieved_goal
        
        self.sparse_reward_type = sparse_reward_type

        obs = self.env.reset()
        obs_dict = self.convert_obs_to_dict(obs)
        
        self.obs_dim = obs_dict['observation'].shape[0]
        self.goal_dim = obs_dict['desired_goal'].shape[0]
        
        self.proprioceptive_only = False

        self.convert_goal_to_reach_object = convert_goal_to_reach_object
        # temporarily commented for aim_train with earl env
        # print('currently, commented dict observation space for aim train with earl env!')
        # self.observation_space = gym.spaces.Dict(
        #     dict(
        #         desired_goal=gym.spaces.Box(
        #             -np.inf, np.inf, shape=obs_dict["achieved_goal"].shape, dtype="float32"
        #         ),
        #         achieved_goal=gym.spaces.Box(
        #             -np.inf, np.inf, shape=obs_dict["achieved_goal"].shape, dtype="float32"
        #         ),
        #         observation=gym.spaces.Box(
        #             -np.inf, np.inf, shape=obs_dict["observation"].shape, dtype="float32"
        #         ),
        #     )
        # )
        
    
    def replace_goal_in_obs(self, obs, goal):
        if 'tabletop' in self.env_name:            
            obs[..., 6:] = goal.copy()
            
        elif 'sawyer' in self.env_name:
            if self.env_name=='sawyer_door':
                if self.env.add_velocity_info=='door':
                    obs[..., 13:] = goal.copy()
                elif self.env.add_velocity_info=='ee_door':
                    obs[..., 10:] = goal.copy()
                else:
                    obs[..., 7:] = goal.copy()
            else:
                obs[..., 7:] = goal.copy()

        else:
            raise NotImplementedError        
        return obs

    def get_achieved_goal_from_pure_obs(self, obs):
        if 'tabletop' in self.env_name:
            assert obs.shape[-1]==6
            achieved_goal = obs[..., :6].copy()            
        elif 'sawyer' in self.env_name:
            if self.env_name=='sawyer_door':
                if self.env.add_velocity_info=='door':
                    assert obs.shape[-1]==13
                elif self.env.add_velocity_info=='ee_door':
                    assert obs.shape[-1]==13                
                else:
                    assert obs.shape[-1]==7
            else:
                assert obs.shape[-1]==7
            achieved_goal = obs[..., :7].copy()
        elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
            assert obs.shape[-1]==25
            achieved_goal = obs[..., 3:6].copy()  # obj pos
        elif self.env_name in ['fetch_reach_ergodic']:
            assert obs.shape[-1]==10
            achieved_goal = obs[..., :3].copy()  # ee pos
        elif self.env_name in ['point_umaze']:
            assert obs.shape[-1]==7
            achieved_goal = obs[..., :2].copy()  # ee pos
        else:
            raise NotImplementedError
        return achieved_goal
    
    def get_fully_concatenated_obs_from_pure_obs(self, obs, desired_goal):
        if 'tabletop' in self.env_name:
            assert obs.shape[-1]==6
            return concatenate(obs, desired_goal)
        elif 'sawyer' in self.env_name:
            assert self.env.add_velocity_info is None
            assert obs.shape[-1]==7
            return concatenate(obs, desired_goal)
        elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
            assert obs.shape[-1]==25
            achieved_goal = self.get_achieved_goal_from_pure_obs(obs)
            return concatenate(obs, achieved_goal, desired_goal)            
        elif self.env_name in ['fetch_reach_ergodic']:
            assert obs.shape[-1]==10
            achieved_goal = self.get_achieved_goal_from_pure_obs(obs)
            return concatenate(obs, achieved_goal, desired_goal)            
        elif self.env_name in ['point_umaze']:
            assert obs.shape[-1]==7
            achieved_goal = self.get_achieved_goal_from_pure_obs(obs)
            return concatenate(obs, achieved_goal, desired_goal)            
        else:
            raise NotImplementedError

    def convert_dict_to_obs(self, obs_dict, batch_ver=False):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic', 'fetch_reach_ergodic', 'point_umaze']:
            return np.concatenate([obs_dict[key] for key in KEY_ORDER], axis = -1)
        else:
            return np.concatenate([obs_dict[key] for key in self.reduced_key_order], axis = -1)
            

    def convert_obs_to_dict(self, obs, batch_ver=False):
        
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        # Currently restricted to FetchEnv
        if 'tabletop' in self.env_name:
            assert obs.shape[-1]==12, 'obs shape is {}'.format(obs.shape)
            return {
                "observation": obs[..., :6] ,
                "achieved_goal": obs[..., :6] ,
                "desired_goal": obs[..., 6:] ,
            }
            
        elif 'sawyer' in self.env_name:
            if self.env_name=='sawyer_door':
                if self.env.add_velocity_info=='door':
                    assert obs.shape[-1]==17, 'obs shape is {}'.format(obs.shape)
                    return {
                        "observation": obs[..., :10] ,
                        "achieved_goal": obs[..., :7] ,
                        "desired_goal": obs[..., 10:] ,
                    }
                elif self.env.add_velocity_info=='ee_door':
                    assert obs.shape[-1]==20, 'obs shape is {}'.format(obs.shape)
                    return {
                        "observation": obs[..., :13] ,
                        "achieved_goal": obs[..., :7] ,
                        "desired_goal": obs[..., 13:] ,
                    }
                else:
                    assert obs.shape[-1]==14, 'obs shape is {}'.format(obs.shape)
                    return {
                        "observation": obs[..., :7] ,
                        "achieved_goal": obs[..., :7] ,
                        "desired_goal": obs[..., 7:] ,
                    }
            else:                    
                assert obs.shape[-1]==14, 'obs shape is {}'.format(obs.shape)
                return {
                    "observation": obs[..., :7] ,
                    "achieved_goal": obs[..., :7] ,
                    "desired_goal": obs[..., 7:] ,
                }

        elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:            
            if self.env.full_state_goal:
                assert obs.shape[-1]==37, 'obs shape is {}'.format(obs.shape)
                return {
                    "observation": obs[..., :25] ,
                    "achieved_goal": obs[..., 25:31] ,
                    "desired_goal": obs[..., 31:37] ,
                }
            else:
                assert obs.shape[-1]==31, 'obs shape is {}'.format(obs.shape)
                return {
                    "observation": obs[..., :25] ,
                    "achieved_goal": obs[..., 25:28] ,
                    "desired_goal": obs[..., 28:31] ,
                }

        elif self.env_name in ['fetch_reach_ergodic']:            
            assert obs.shape[-1]==16, 'obs shape is {}'.format(obs.shape)
            return {
                "observation": obs[..., :10] ,
                "achieved_goal": obs[..., 10:13] ,
                "desired_goal": obs[..., 13:16] ,
            }

        elif self.env_name in ['point_umaze']:            
            assert obs.shape[-1]==11, 'obs shape is {}'.format(obs.shape)
            return {
                "observation": obs[..., :7] ,
                "achieved_goal": obs[..., 7:9] ,
                "desired_goal": obs[..., 9:11] ,
            }

        else:
            raise NotImplementedError

    def is_successful_deviating_initial_state(self, obs):
        if self.env_name=='sawyer_door':
            return np.linalg.norm(obs[..., :7] - self.env.init_state[..., :7], axis =-1) > 0.02
            # if you want consider only hand
            # return np.linalg.norm(obs[..., :3] - self.env.init_state[..., :3], axis =-1) > 0.02
        
        elif self.env_name=='tabletop':
            return np.linalg.norm(obs[..., :4] - self.env.init_state[..., :4], axis =-1) > 0.2
            # if you want consider only hand
            # return np.linalg.norm(obs[..., :2] - self.env.init_state[..., :2], axis =-1) > 0.2

        else:
            raise NotImplementedError
    
    def is_different_init_state_and_goal(self, obs):
        if self.env_name=='sawyer_door':
            assert self.env.add_velocity_info is None
            return np.linalg.norm(obs[..., 7:14] - self.env.init_state[..., :7], axis =-1) > 0.02
            # if you want consider only hand
            # return np.linalg.norm(obs[..., 7:10] - self.env.init_state[..., :3], axis =-1) > 0.02
            
        elif self.env_name=='tabletop':
            return np.linalg.norm(obs[..., 6:10] - self.env.init_state[..., :4], axis =-1) > 0.2
            # if you want consider only hand
            # return np.linalg.norm(obs[..., 6:8] - self.env.init_state[..., :2], axis =-1) > 0.2

        else:
            raise NotImplementedError

    # for EARL env
    # used when HER relabeling
    def compute_reward(self, obs, proprioceptive_only=False):        
        # Assume sparse reward!
        if self.sparse_reward_type=='positive':            
            return (self.is_successful(obs=obs, proprioceptive_only=proprioceptive_only)).astype(np.float)
        elif self.sparse_reward_type=='negative':
            return (self.is_successful(obs=obs, proprioceptive_only=proprioceptive_only)).astype(np.float)-1.0
    
    # used when HER relabeling
    def is_successful(self, obs, proprioceptive_only=False):
        if getattr(self.env, 'convert', False): # For ConvertGoalToReachObjectEnv
            return self.env.is_successful(obs)
        else:
            # NOTE : Assume obs : [ag, dg]            
            if proprioceptive_only: # consider reach only even though the object exists
                if self.env_name in ['sawyer_door']:                    
                    assert (obs.shape[-1]==14 or obs.shape[-1]==17 or obs.shape[-1]==20)
                    return np.linalg.norm(obs[..., :3] - obs[..., -7:-4], axis =-1) <= 0.05
                elif self.env_name=='tabletop_manipulation':
                    assert obs.shape[-1]==12
                    return np.linalg.norm(obs[..., :2] - obs[..., 6:8], axis =-1) <= 0.2
                elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
                    if self.env.full_state_goal: # g : [grip(3), obj(3)]
                        assert obs.shape[-1]==12
                        return np.linalg.norm(obs[..., :3] - obs[..., -6:-3], axis =-1) <= 0.05
                    else:  # g : [grip(3)]
                        assert obs.shape[-1]==6
                        return np.linalg.norm(obs[..., -6:-3] - obs[..., -3:], axis =-1) <= 0.05
                elif self.env_name in ['fetch_reach_ergodic']:
                    assert not self.env.full_state_goal
                    assert obs.shape[-1]==6
                    return np.linalg.norm(obs[..., -6:-3] - obs[..., -3:], axis =-1) <= 0.05
                elif self.env_name in ['point_umaze']:
                    assert obs.shape[-1]==4
                    return np.linalg.norm(obs[..., -4:-2] - obs[..., -2:], axis =-1) <= 0.6
                else:
                    raise NotImplementedError
            else: # original forward task
                if self.env_name=='sawyer_door':
                    return np.linalg.norm(obs[..., 4:7] - obs[..., -3:], axis =-1) <= 0.02
                elif self.env_name=='tabletop_manipulation':
                    return np.linalg.norm(obs[..., :4] - obs[..., 6:-2], axis =-1) <= 0.2
                elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
                    if self.env.full_state_goal: # NOTE: currently, we do not consider grip & object both! only object!                        
                        assert obs.shape[-1]==12
                        return np.linalg.norm(obs[..., 3:6] - obs[..., -3:], axis =-1) <= 0.05
                    else:
                        assert obs.shape[-1]==6
                        return np.linalg.norm(obs[..., -6:-3] - obs[..., -3:], axis =-1) <= 0.05
                elif self.env_name in ['fetch_reach_ergodic']:
                    assert not self.env.full_state_goal
                    assert obs.shape[-1]==6
                    return np.linalg.norm(obs[..., -6:-3] - obs[..., -3:], axis =-1) <= 0.05
                elif self.env_name in ['point_umaze']:
                    assert obs.shape[-1]==4
                    return np.linalg.norm(obs[..., -4:-2] - obs[..., -2:], axis =-1) <= 0.6
                else:
                    raise NotImplementedError
        
    
    def get_hand_pos(self, obs):
        if self.env_name=='sawyer_door':
            return obs[..., :3]
        elif self.env_name=='tabletop_manipulation':
            return obs[..., :2]
        elif 'Fetch' in self.env_name:
            return obs[..., :3]
        else:
            raise NotImplementedError

    def __getattr__(self, attrname):
        return getattr(self.env, attrname)


############### for AIM

from collections import OrderedDict
import numpy as np
from gym import spaces
KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']


class HERGoalEnvWrapper(object):
    """
    A wrapper that allow to use dict observation space (coming from GoalEnv) with
    the RL algorithms.
    It assumes that all the spaces of the dict space are of the same type.

    :param env: (gym.GoalEnv)
    """

    def __init__(self, env, env_name = None):
        super(HERGoalEnvWrapper, self).__init__()
        self.env = env
        self.env_name = env_name
        self.metadata = self.env.metadata
        self.action_space = env.action_space
        self.proprioceptive_only = False

        if env_name in ['point_umaze']: # due to the different observation space format
            self.spaces = list(env.observation_space.values())
            space_types = [type(env.observation_space[key]) for key in KEY_ORDER]
        else:
            self.spaces = list(env.observation_space.spaces.values())
            # Check that all spaces are of the same type
            # (current limitation of the wrapper)
            space_types = [type(env.observation_space.spaces[key]) for key in KEY_ORDER]
        assert len(set(space_types)) == 1, "The spaces for goal and observation"\
                                           " must be of the same type"

        if isinstance(self.spaces[0], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            if env_name in ['point_umaze']:
                goal_space_shape = env.observation_space['achieved_goal'].shape
                self.obs_dim = env.observation_space['observation'].shape[0]
            else:
                goal_space_shape = env.observation_space.spaces['achieved_goal'].shape
                self.obs_dim = env.observation_space.spaces['observation'].shape[0]
            self.goal_dim = goal_space_shape[0]

            if len(goal_space_shape) == 2:
                assert goal_space_shape[1] == 1, "Only 1D observation spaces are supported yet"
            else:
                assert len(goal_space_shape) == 1, "Only 1D observation spaces are supported yet"

        if isinstance(self.spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + 2 * self.goal_dim
            self.observation_space = spaces.MultiBinary(total_dim)

        elif isinstance(self.spaces[0], spaces.Box):
            lows = np.concatenate([space.low for space in self.spaces])
            highs = np.concatenate([space.high for space in self.spaces])
            self.observation_space = spaces.Box(lows, highs, dtype=np.float32)

        elif isinstance(self.spaces[0], spaces.Discrete):
            dimensions = [env.observation_space.spaces[key].n for key in KEY_ORDER]
            self.observation_space = spaces.MultiDiscrete(dimensions)

        else:
            raise NotImplementedError("{} space is not supported".format(type(self.spaces[0])))

    def convert_dict_to_obs(self, obs_dict):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in KEY_ORDER])
        return np.concatenate([obs_dict[key] for key in KEY_ORDER], axis =-1)

    def convert_obs_to_dict(self, observations):
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        return OrderedDict([
            ('observation', observations[..., :self.obs_dim]),
            ('achieved_goal', observations[..., self.obs_dim:self.obs_dim + self.goal_dim]),
            ('desired_goal', observations[..., self.obs_dim + self.goal_dim:]),
        ])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.convert_dict_to_obs(obs), reward, done, info

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self, *args, **kwargs):
        return self.convert_dict_to_obs(self.env.reset(*args, **kwargs))

    def compute_reward(self, achieved_goal, desired_goal, *args, **kwargs): # info=None,
        return self.env.compute_reward(achieved_goal, desired_goal, *args, **kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()
    
    def set_proprioceptive_only(self, proprioceptive_only): # should be called after forward, backward direction is changed
        self.proprioceptive_only = proprioceptive_only
        self.env.set_proprioceptive_only(proprioceptive_only)
    
    
    def is_successful(self, obs, proprioceptive_only = False):
        if getattr(self.env, 'convert', False): # For ConvertGoalToReachObjectEnv
            raise NotImplementedError('Currently, interactive env with convert is not used for non EARL env (ex. Fetch, Ant) with HERGoalEnvWrapper. \
                Later, Fetch can be used with convert. Or Integrate HERGoalEnvWrapper & WrapToGoalEnv')
            return self.env.is_successful(obs)
        else: # used in eval_env or hgg or full_obs_success_check! # NOTE: Only considering forward task goal (not proprioceptive_only case!)
            if self.proprioceptive_only or proprioceptive_only: # consider only reach
                if self.env_name in ['sawyer_door', 'tabletop_manipulation']:                    
                    raise NotImplementedError('EARL envs have this method, so, this wrapper will not be used')
                elif self.env_name in ['fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
                    if self.env_name == 'fetch_reach_ergodic':
                        assert not self.env.full_state_goal
                        assert obs.shape[-1]==16, 'assume obs, ag, dg concatenated'
                        return np.linalg.norm(obs[..., -6:-3] - obs[..., -3:], axis =-1) <= 0.05
                    else: # consider only gripper
                        if self.env.full_state_goal:                        
                            assert obs.shape[-1]==37, 'assume obs, ag, dg concatenated'
                            return np.linalg.norm(obs[..., -12:-9] - obs[..., -6:-3], axis =-1) <= 0.05
                        else:
                            assert obs.shape[-1]==31, 'assume obs, ag, dg concatenated'
                            return np.linalg.norm(obs[..., -6:-3] - obs[..., -3:], axis =-1) <= 0.05

                elif self.env_name in ['point_umaze']:
                    assert obs.shape[-1]==11, 'assume obs, ag, dg concatenated'
                    return np.linalg.norm(obs[..., -4:-2] - obs[..., -2:], axis =-1) <= 0.6
                else:
                    raise NotImplementedError            
            else:                
                if self.env_name=='sawyer_door':
                    raise NotImplementedError('EARL envs have this method, so, this wrapper will not be used')
                    return np.linalg.norm(obs[..., 4:7] - obs[..., 11:14], axis =-1) <= 0.02
                elif self.env_name=='tabletop_manipulation':
                    raise NotImplementedError('EARL envs have this method, so, this wrapper will not be used')
                    return np.linalg.norm(obs[..., :4] - obs[..., 6:-2], axis =-1) <= 0.2
                elif 'Fetch' in self.env_name:
                    return np.linalg.norm(obs[..., -6:-3] - obs[..., -3:], axis =-1) <= 0.05
                elif self.env_name in ['fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
                    if not self.env.full_state_goal:                    
                        if self.env_name == 'fetch_reach_ergodic':
                            assert obs.shape[-1]==16, 'assume obs, ag, dg concatenated'
                        elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:                                        
                            assert obs.shape[-1]==31, 'assume obs, ag, dg concatenated'
                        return np.linalg.norm(obs[..., -6:-3] - obs[..., -3:], axis =-1) <= 0.05

                    else: # only consider object
                        if self.env_name == 'fetch_reach_ergodic':
                            raise NotImplementedError
                        elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:                                        
                            assert obs.shape[-1]==37, 'assume obs, ag, dg concatenated'
                        return np.linalg.norm(obs[..., -9:-6] - obs[..., -3:], axis =-1) <= 0.05

                elif self.env_name in ['point_umaze']:
                    assert obs.shape[-1]==11, 'assume obs, ag, dg concatenated'
                    return np.linalg.norm(obs[..., -4:-2] - obs[..., -2:], axis =-1) <= 0.6
                else:
                    raise NotImplementedError
    
    
    def get_hand_pos(self, obs):
        if self.env_name=='sawyer_door':
            return obs[..., :3]
        elif self.env_name=='tabletop_manipulation':
            return obs[..., :2]
        elif 'Fetch' in self.env_name:
            return obs[..., :3]
        else:
            raise NotImplementedError

   

    def __getattr__(self, attrname):
        return getattr(self.env, attrname)

import copy
class RewardOffsetWrapper(gym.Wrapper):
    """
    Reward Offeset Wrapper.
    """
    def __init__(self, env, reward_offset=0.0):
        super(RewardOffsetWrapper, self).__init__(env)
        self.reward_offset = reward_offset
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)        
        reward += self.reward_offset
        return obs, reward, done, info
    
    def compute_reward(self, achieved_goal, desired_goal, *args, **kwargs): # info=None        
        reward = self.env.compute_reward(achieved_goal, desired_goal, *args, **kwargs) # info
        return reward + self.reward_offset

    def __getattr__(self, attrname):
        return getattr(self.env, attrname)



class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0, earl_env = False, antmaze_env = False):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset
        self.earl_env = earl_env
        self.antmaze_env = antmaze_env
        # if earl_env:
        #     assert reward_offset==0.0, 'assume earl (sawyer, tabletop) outputs 0,1 sparse reward'

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.earl_env:
            # currently, done is earl done 
            info.update({'earl_done' : copy.deepcopy(done)})
            done = info.get('ConvertGoalToReachObjectEnv_done', done) 
            
        done = done or info.get('is_success', False) # also used in fetch_ergodic, pointumaze
        if self.earl_env:
            done  = done or self.env.is_successful(obs)            
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, *args, **kwargs): # info=None
        # used in episodic aim (with HERGoalEnvWrapper)        
        if self.antmaze_env:            
            reward = self.env.compute_reward(achieved_goal, desired_goal, *args, **kwargs)
        else:
            reward = self.env.compute_reward(achieved_goal, desired_goal, *args, **kwargs) # info
        return reward + self.reward_offset
    
    def __getattr__(self, attrname):
        return getattr(self.env, attrname)



class RewardChangeWrapperEnv(object): 
    '''
    NOTE : Reward function change according to user's intention. This wrapper should be used right after gym.make!
    '''
    def __init__(self, env, env_name, *args, **kwargs):        
        self.env = env                        
        self.env_name = env_name
        self.proprioceptive_only=False

    def set_proprioceptive_only(self, proprioceptive_only): # should be called after forward, backward direction is changed
        self.proprioceptive_only = proprioceptive_only
        self.env.set_proprioceptive_only(proprioceptive_only)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if self.env_name in ['sawyer_door']: #[ee(3), grip(1), obj(3)]
            assert (next_obs.shape[-1]==14 or next_obs.shape[-1]==17 or next_obs.shape[-1]==20)
            # reward [0, 1]            
            if self.proprioceptive_only:
                threshold = 0.05 # gripper
                reward = (np.linalg.norm(next_obs[..., :3]-next_obs[..., -7:-4], axis =-1) < threshold).astype(np.float32)
            else:
                threshold = 0.02 if self.env_name == 'sawyer_door' else self.env.TARGET_RADIUS            
                reward = (np.linalg.norm(next_obs[..., 4:7]-next_obs[..., -3:], axis =-1) < threshold).astype(np.float32)
        
        elif self.env_name=='tabletop_manipulation': #[ee(2), obj(2), grip_state(2)]
            assert next_obs.shape[-1]==12
            # reward [0, 1]
            if self.proprioceptive_only:            
                reward = (np.linalg.norm(next_obs[..., :2]-next_obs[..., 6:8], axis =-1) < 0.2).astype(np.float32)
            else:
                reward = (np.linalg.norm(next_obs[..., :4]-next_obs[..., 6:10], axis =-1) < 0.2).astype(np.float32)
            
        elif self.env_name in ['fetch_pickandplace_ergodic', 'fetch_push_ergodic' ]: #[grip_pos(3), object_pos(3), ...]              
            assert next_obs['observation'].shape[-1]==25
            # reward [-1, 0]
            if self.env.full_state_goal:
                assert next_obs['achieved_goal'].shape[-1]==6
                if self.proprioceptive_only: # gripper
                    reward = -(np.linalg.norm(next_obs['achieved_goal'][..., :3]-next_obs['desired_goal'][..., :3], axis =-1) > 0.05).astype(np.float32)
                else:
                    reward = -(np.linalg.norm(next_obs['achieved_goal'][..., -3:]-next_obs['desired_goal'][..., -3:], axis =-1) > 0.05).astype(np.float32)
            else:
                assert next_obs['achieved_goal'].shape[-1]==3
                reward = -(np.linalg.norm(next_obs['achieved_goal'][..., -3:]-next_obs['desired_goal'][..., -3:], axis =-1) > 0.05).astype(np.float32)
        else: # no object env -> use the given sparse reward as it is
            pass

        return next_obs, reward, done, info

    def compute_reward(self, obs=None, info=None):
        raise NotImplementedError

    def __getattr__(self, attrname):
        return getattr(self.env, attrname)
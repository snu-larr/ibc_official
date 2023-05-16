import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import os
import random

class ForwardBackwardReplayBufferWrapper(object):
    def __init__(self, forward_buffer, backward_buffer):
        self.forward_buffer = forward_buffer    
        self.backward_buffer = backward_buffer
        self.option = 'forward'

    def __getattr__(self, name):
        if self.option =='forward':
            return getattr(self.forward_buffer, name)
        elif self.option =='backward':
            return getattr(self.backward_buffer, name)
    
    def __len__(self):
        if self.option=='forward':
            return len(self.forward_buffer)
        elif self.option=='backward':
            return len(self.backward_buffer)

    def set_option(self, option):
        self.option = option
    
    def sample(self, batch_size, discount, proprioceptive_only = False):
        if self.option=='forward':
            forward_samples = self.forward_buffer.sample(int(batch_size/2), discount, proprioceptive_only=proprioceptive_only)
            return forward_samples
        
        elif self.option=='backward':
            backward_samples = self.backward_buffer.sample(batch_size, discount, proprioceptive_only=proprioceptive_only)
            return backward_samples

    
    def sample_goal_examples(self, batch_size, sample_only_state = True):
        if self.option=='forward':    
            return self.forward_buffer.sample_goal_examples(batch_size, sample_only_state)
        elif self.option=='backawrd':
            return self.backward_buffer.sample_initial_examples(batch_size, sample_only_state)
            



class ReplayBuffer(object):
    state_idx_dict = {'tabletop_manipulation' : 6,
                      'sawyer_door' : 7,
                      
                      }

    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, 
                traj_length = None, sample_type=None, env_name = None, success_labeling = False,                 
                
                ):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity
        self.device = device
        
        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.last_save = 0

        # for saving trajectory
        self.sample_type = sample_type
        self.env_name = env_name
        self.traj_length = traj_length
        if traj_length is not None:            
            self.trajwise_capacity = int(capacity/traj_length)
            self.observes_traj = np.empty((self.trajwise_capacity, traj_length, *obs_shape), dtype=np.float32)        
            self.observes_traj_idx = 0
            self.trajwise_full = False
            self.episode_observes = []
        

        self.success_labeling = success_labeling 
        if success_labeling:
            self.goal_successes = np.empty((capacity, 1), dtype= np.float32)
        
        
    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        
        if self.success_labeling:
            assert reward==0 or reward ==1, 'assume sparse (0,1) reward'
            np.copyto(self.goal_successes[self.idx], reward)

        # if self.traj_length is not None:
        #     self.episode_observes.append(obs)
        #     if last_timestep:                
        #         np.copyto(self.observes_traj[self.observes_traj_idx], np.stack(self.episode_observes, axis =0)) #[ts, dim]
        #         self.observes_traj_idx = (self.observes_traj_idx + 1) % self.trajwise_capacity
        #         self.trajwise_full = self.trajwise_full or self.observes_traj_idx==0
        #         self.episode_observes = []
    

    # should be called outside
    def add_trajectory(self, episode_observes):
        assert type(episode_observes) is list
        if self.traj_length is not None:
            self.episode_observes = episode_observes #.append(obs)            
            np.copyto(self.observes_traj[self.observes_traj_idx], np.stack(self.episode_observes, axis =0)) #[ts, dim]
            self.observes_traj_idx = (self.observes_traj_idx + 1) % self.trajwise_capacity
            self.trajwise_full = self.trajwise_full or self.observes_traj_idx==0
            self.episode_observes = []
        

    def sample_trajwise_observation(self, batch_size, sample_type=None):
        idxs = np.random.randint(0,
                                 self.trajwise_capacity if self.trajwise_full else self.observes_traj_idx,
                                 size=batch_size)
        if sample_type is None :
            sample_type = self.sample_type

        if sample_type=='only_state':
            obses = torch.as_tensor(self.observes_traj[idxs, :, :self.state_idx_dict[self.env_name]], device=self.device).float() #[bs, ts, dim]
        elif sample_type=='with_initial_state':
            pure_obs = self.observes_traj[idxs, :self.state_idx_dict[self.env_name]]
            init_state = self.observes_traj[idxs, -self.state_idx_dict[self.env_name]:]
            obses = torch.as_tensor(np.concatenate([pure_obs, init_state], axis =-1), device=self.device).float() #[bs, ts, dim]            
        else:
            obses = torch.as_tensor(self.observes_traj[idxs], device=self.device).float() #[bs, ts, dim]
        
        return obses

    def get_random_indices(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        return idxs

    def sample(self, batch_size, discount, idxs = None):
        # idxs = np.random.randint(0,
        #                          self.capacity if self.full else self.idx,
        #                          size=batch_size)
        if idxs is None:
            idxs = self.get_random_indices(batch_size)
        
        
        if self.sample_type=='only_state':
            # assume original state is concatenated one.
            obses = torch.as_tensor(self.obses[idxs, :self.state_idx_dict[self.env_name]], device=self.device).float()
            next_obses = torch.as_tensor(self.next_obses[idxs, :self.state_idx_dict[self.env_name]],
                                        device=self.device).float()
        elif self.sample_type=='with_initial_state':
            # assume original state is concatenated one.
            pure_obs = self.obses[idxs, :self.state_idx_dict[self.env_name]]
            init_state = self.obses[idxs, -self.state_idx_dict[self.env_name]:]
            pure_next_obs = self.next_obses[idxs, :self.state_idx_dict[self.env_name]]
            init_next_state = self.next_obses[idxs, -self.state_idx_dict[self.env_name]:]
            obses = torch.as_tensor(np.concatenate([pure_obs, init_state], axis =-1), device=self.device).float()
            next_obses = torch.as_tensor(np.concatenate([pure_next_obs, init_next_state], axis =-1), device=self.device).float()
        else:
            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            next_obses = torch.as_tensor(self.next_obses[idxs],
                                        device=self.device).float()
                
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        discounts = np.ones((idxs.shape[0], 1), dtype=np.float32) * discount
        discounts = torch.as_tensor(discounts, device=self.device)                
        dones = torch.as_tensor(~self.not_dones[idxs].astype(bool), device=self.device).float()


        return obses, actions, rewards, next_obses, discounts, dones

    
    def sample_all_data(self):        
        return dict(observation=self.obses,
                    action = self.actions,
                    reward = self.rewards,
                    next_observation=self.next_obses,
                    not_done = self.not_dones,
                    )

    def sample_data_collected_until_now(self, discount):
        if self.full:
            return torch.as_tensor(self.obses, device=self.device),\
                    torch.as_tensor(self.actions, device=self.device),\
                    torch.as_tensor(self.rewards, device=self.device),\
                    torch.as_tensor(self.next_obses, device=self.device),\
                    torch.as_tensor(np.ones((self.obses.shape[0], 1), dtype=np.float32), device=self.device) * discount,\
                    torch.as_tensor(self.not_dones, device=self.device)
                    
                    
        else:    
            return torch.as_tensor(self.obses[:self.idx], device=self.device),\
                    torch.as_tensor(self.actions[:self.idx], device=self.device),\
                    torch.as_tensor(self.rewards[:self.idx], device=self.device),\
                    torch.as_tensor(self.next_obses[:self.idx], device=self.device),\
                    torch.as_tensor(np.ones((self.obses[:self.idx].shape[0], 1), dtype=np.float32), device=self.device) * discount,\
                    torch.as_tensor(self.not_dones[:self.idx], device=self.device)
                    
                    

    def sample_without_relabeling(self, batch_size, discount, sample_only_state = True):
        # should be called in forward gcrl buffer ()
        obses, actions, rewards, next_obses, discounts, dones = self.sample(batch_size, discount)
        if sample_only_state:
            obses = obses[:, :self.state_idx_dict[self.env_name]]
            next_obses = next_obses[:, :self.state_idx_dict[self.env_name]]
        
        return obses, actions, rewards, next_obses, discounts, dones
    
    def sample_goal_examples(self, batch_size, sample_only_state = True):
        
        success_goal_indices = np.where(self.goal_successes==1)[0]
        
        if success_goal_indices.shape[0]<=10: # If less than 10(just heuristic), return None (do not update the discriminator)
            return None

        idxs = np.random.randint(0, success_goal_indices.shape[0], size=batch_size)
        goal_examples = self.obses[success_goal_indices[idxs]]
                
        goal_examples = torch.as_tensor(goal_examples, device=self.device).float()
        
        if sample_only_state:
            goal_examples = goal_examples[:, :self.state_idx_dict[self.env_name]]

        return goal_examples

    def sample_initial_examples(self, batch_size, sample_only_state = True):
        idxs = np.random.randint(0, self.initial_examples.shape[0], size=batch_size)
        initial_examples = torch.as_tensor(self.initial_examples[idxs]).float()
        assert sample_only_state
        
        if sample_only_state:
            initial_examples = initial_examples[:, :self.state_idx_dict[self.env_name]]

        return initial_examples

import copy
from enum import Enum

import numpy as np


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """
    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # Select a goal that was achieved
    # at some point in the training procedure
    # (and that is present in the replay buffer)
    RANDOM = 3


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM
}


class HindsightExperienceReplayWrapper(object):
    """
    Wrapper around a replay buffer in order to use HER.
    This implementation is inspired by to the one found in https://github.com/NervanaSystems/coach/.

    :param replay_buffer: (ReplayBuffer)
    :param n_sampled_goal: (int) The number of artificial transitions to generate for each actual transition
    :param goal_selection_strategy: (GoalSelectionStrategy) The method that will be used to generate
        the goals for the artificial transitions.
    :param wrapped_env: (BaselineHERGoalEnvWrapper) the GoalEnv wrapped using BaselineHERGoalEnvWrapper,
        that enables to convert observation to dict, and vice versa
    """

    def __init__(self, replay_buffer, n_sampled_goal, goal_selection_strategy, wrapped_env, env_name):
        super(HindsightExperienceReplayWrapper, self).__init__()

        assert isinstance(goal_selection_strategy, GoalSelectionStrategy), "Invalid goal selection strategy," \
                                                                           "please use one of {}".format(
            list(GoalSelectionStrategy))
        
        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.env = wrapped_env
        self.env_name = env_name
        # Buffer for storing transitions of the current episode
        self.episode_transitions = []
        self.replay_buffer = replay_buffer

    def add(self, obs_t, action, reward, obs_tp1, done, last_timestep=False):
        """
        add a new transition to the buffer

        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        """
        assert self.replay_buffer is not None
        # Update current episode buffer
        self.episode_transitions.append(copy.deepcopy((obs_t, action, reward, obs_tp1, done)))
        if last_timestep:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []

    def sample(self, *args, **kwargs):
        return self.replay_buffer.sample(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.replay_buffer, name)
    
    
    # def sample_path(self, *args, **kwargs):
    #     return self.replay_buffer.sample_path(*args, **kwargs)

    # def can_sample(self, n_samples):
    #     """
    #     Check if n_samples samples can be sampled
    #     from the buffer.

    #     :param n_samples: (int)
    #     :return: (bool)
    #     """
    #     return self.replay_buffer.can_sample(n_samples)

    def __len__(self):
        return len(self.replay_buffer)


    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling strategy.

        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random goal achieved, from the entire replay buffer
            selected_idx = np.random.choice(np.arange(len(self.replay_buffer)))
            selected_transition = self.replay_buffer.storage[selected_idx]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal']

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.

        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        for transition_idx, transition in enumerate(self.episode_transitions):

            obs_t, action, reward, obs_tp1, done = transition

            # Add to the replay buffer
            self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(self.episode_transitions) - 1 and
                    self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                break

            # Sampled n goals per transition, where n is `n_sampled_goal`
            # this is called k in the paper
            sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
            # For each sampled goals, store a new transition
            for goal in sampled_goals:
                # Copy transition to avoid modifying the original one
                obs, action, reward, next_obs, done = copy.deepcopy(transition)

                # Convert concatenated obs to dict, so we can update the goals
                obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))

                # Update the desired goal in the transition
                obs_dict['desired_goal'] = goal
                next_obs_dict['desired_goal'] = goal

                # Update the reward according to the new desired goal
                # reward = self.env.compute_reward(next_obs_dict['achieved_goal'], goal, None)
                # for EARL env
                if self.env_name =='sawyer_door':
                    reward, obj_to_target, hand_in_place = self.env.compute_reward(np.concatenate([next_obs_dict['achieved_goal'], goal], axis =-1))    
                else:
                    reward = self.env.compute_reward(np.concatenate([next_obs_dict['achieved_goal'], goal], axis =-1))


                # Can we use achieved_goal == desired_goal?
                done = False

                # Transform back to ndarrays
                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                # Add artificial transition to the replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)

    # def _store_path(self):
    #     raise NotImplementedError

    def sample_all_data(self):
        return self.replay_buffer.sample_all_data()




class HindsightExperienceReplayWrapperVer2(object):
    """
    Wrapper around a replay buffer in order to use HER with memory efficiency.
    Sample relabeled batches when sampling method is called.    
    """

    def __init__(self, replay_buffer, n_sampled_goal, wrapped_env, env_name, consider_done_true = False):
        # super(HindsightExperienceReplayWrapperVer2, self).__init__()
        self.n_sampled_goal = n_sampled_goal
        self.env = wrapped_env
        self.env_name = env_name
        # Buffer for storing transitions of the current episode
        self.episode_transitions = []
        self.replay_buffer = replay_buffer
        self._idx_to_future_obs_idx = [None] * self.replay_buffer.capacity

        # for done on success
        self.consider_done_true = consider_done_true

    def add(self, obs_t, action, reward, obs_tp1, done, last_timestep=False):
        """
        add a new transition to the buffer

        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        """
        assert self.replay_buffer is not None
        # Update current episode buffer
        self.episode_transitions.append(copy.deepcopy((obs_t, action, reward, obs_tp1, done)))
        if last_timestep:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []

    def sample(self, batch_size, discount, indices=None, proprioceptive_only=False):
        if indices is None:
            indices = self.replay_buffer.get_random_indices(batch_size) 
        
        num_rollout_goals = int(batch_size*1/(self.n_sampled_goal+1)) # batch_size*0.2
        num_future_goals = batch_size - num_rollout_goals # batch_size*0.8
        
        #TODO:should know indices used for sampling
        obses, actions, rewards, next_obses, discounts, dones = self.replay_buffer.sample(None, discount, idxs = indices) # batch_size*0.2
        if self.replay_buffer.sample_type in ['only_state', 'with_initial_state'] : #.sample_only_state:
            # do not relabel
            return obses, actions, rewards, next_obses, discounts

        sample_torch_data = False
        if torch.is_tensor(obses):
            # convert from torch to numpy
            sample_torch_data = True            
            obses = obses.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
            rewards = rewards.detach().cpu().numpy()
            next_obses = next_obses.detach().cpu().numpy()
            discounts = discounts.detach().cpu().numpy()
            dones = dones.detach().cpu().numpy()


        obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obses, next_obses)) #TODO: convert should address the batch inputs
        
        if num_future_goals > 0:
            future_indices = indices[-num_future_goals:]
            possible_future_obs_lens = np.array([
                len(self._idx_to_future_obs_idx[i]) for i in future_indices
            ])
            next_obs_idxs = (
                np.random.random(num_future_goals) * possible_future_obs_lens
            ).astype(np.int)
            future_obs_idxs = np.array([
                self._idx_to_future_obs_idx[ids][next_obs_idxs[i]] if self._idx_to_future_obs_idx[ids].shape[0]!=0 else ids # original next_obs idx
                for i, ids in enumerate(future_indices) 
            ]) # idx is global idx in buffer
            assert future_obs_idxs.shape[0]==future_indices.shape[0]
            future_next_obses = self.replay_buffer.next_obses[future_obs_idxs].copy() #[num_future_goals, dim]
            future_next_obses_dict = self.env.convert_obs_to_dict(future_next_obses) #TODO: convert should address the batch inputs
            goal = future_next_obses_dict['achieved_goal'] #[num_future_goals, dim]
            obs_dict['desired_goal'][-num_future_goals:] = goal
            next_obs_dict['desired_goal'][-num_future_goals:] = goal
            
            if self.env_name =='sawyer_door':
                relabeled_reward = self.env.compute_reward(np.concatenate([next_obs_dict['achieved_goal'][-num_future_goals:], goal], axis =-1), proprioceptive_only=proprioceptive_only)    
            elif 'Fetch' in self.env_name:
                relabeled_reward = self.env.compute_reward(next_obs_dict['achieved_goal'][-num_future_goals:], goal, None) # exclude for episodic test proprioceptive_only=proprioceptive_only
            else:
                relabeled_reward = self.env.compute_reward(np.concatenate([next_obs_dict['achieved_goal'][-num_future_goals:], goal], axis =-1), proprioceptive_only=proprioceptive_only)

            
            # Transform back to ndarrays
            relabeled_obs, relabeled_next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict)) #[batch_size]


            obses = relabeled_obs
            next_obses = relabeled_next_obs
            rewards[-num_future_goals:] = relabeled_reward[:, None] #[num_future_goals] -> [num_future_goals,1]
             
            # correspond to done_on_success -> It is used when RL agent utilize done in updating
            # That is, it is used when config consider_done_true_in_critic=true
            if self.consider_done_true:
                if np.min(rewards)==-1.: # (-1,0) sparse
                    dones = rewards + 1. # done = True at reward 0 (success)
                else: # (0,1) sparse
                    dones = np.copy(rewards) # done = True at reward 1 (success)

        if sample_torch_data:
            # re-convert from numpy to torch
            obses = torch.as_tensor(obses, device=self.replay_buffer.device).float()
            actions = torch.as_tensor(actions, device=self.replay_buffer.device).float()
            rewards = torch.as_tensor(rewards, device=self.replay_buffer.device).float()
            discounts = torch.as_tensor(discounts, device=self.replay_buffer.device).float()
            next_obses = torch.as_tensor(next_obses, device=self.replay_buffer.device).float()
            dones = torch.as_tensor(dones, device=self.replay_buffer.device).float()
            
        return obses, actions, rewards, next_obses, discounts, dones
        
    
    def __getattr__(self, name):
        return getattr(self.replay_buffer, name)
    

    def __len__(self):
        return len(self.replay_buffer)


    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        episode_length = len(self.episode_transitions)

        for transition_idx, transition in enumerate(self.episode_transitions):

            obs_t, action, reward, obs_tp1, done = transition
            
            # if transition_idx+1 == episode_length -> idx_to_future_obs_idx[current_transition_idx] = np.array([]) (empty)
                
            # TODO : should consider when buffer is full  
            current_transition_idx = copy.deepcopy(self.replay_buffer.idx)
            remained_timesteps_in_current_episode = episode_length - transition_idx -1
            
            if current_transition_idx+1+remained_timesteps_in_current_episode >self.replay_buffer.capacity:
                # should consider when buffer is full  
                # if current_transition_idx ==999999, -> rear : empty , then next current_transition_idx ==0, escape if lines.
                # if current_transition_idx+1+remained_timesteps_in_current_episode ==1000001, (first time if lines is true), 
                future_obs_indices_rear = np.arange(current_transition_idx+1, self.replay_buffer.capacity) #e.g. [10]
                future_obs_indices_front = np.arange(0, remained_timesteps_in_current_episode - future_obs_indices_rear.shape[0]) # [39]

                self._idx_to_future_obs_idx[current_transition_idx] = np.concatenate([future_obs_indices_rear, future_obs_indices_front], axis=0)
            else:
                self._idx_to_future_obs_idx[current_transition_idx] = np.arange(current_transition_idx+1, current_transition_idx+1+remained_timesteps_in_current_episode)
            
            # Add to the replay buffer
            self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)
            
            
            

    def sample_all_data(self):
        return self.replay_buffer.sample_all_data()



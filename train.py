import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'

import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math
import pickle as pkl
import sys
import time
from queue import Queue
import numpy as np
import hydra
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer, HindsightExperienceReplayWrapperVer2
from video import VideoRecorder
from hgg.hgg import goal_distance
from ibc import add_noise_to_goal
torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.model_dir = utils.make_dir(self.work_dir, 'model')
        
        self.forward_buffer_dir = utils.make_dir(self.work_dir, 'forward_buffer')
        self.backward_buffer_dir = utils.make_dir(self.work_dir, 'backward_buffer')
        self.cfg = cfg
        
        max_episode_timesteps_dict = {'sawyer_door' : 200 if cfg.done_on_success else 100, # to prevent collecting too many non-moving transition data at goal state                                    
                                    'tabletop_manipulation' : 100 if cfg.done_on_success else 50, 
                                    'fetch_reach_ergodic' : 50,
                                    'fetch_push_ergodic' : 50,
                                    'fetch_pickandplace_ergodic' : 50,
                                    'point_umaze' : 100 if cfg.done_on_success else 50,                                     
                                    }

        num_train_steps_dict = {'sawyer_door' : int(2e6),                                
                                'tabletop_manipulation' : int(2e6),
                                'fetch_reach_ergodic' : int(1e6),
                                'fetch_push_ergodic' : int(2e6),
                                'fetch_pickandplace_ergodic' : int(2e6),
                                'point_umaze' : int(1e6),                                
                                }
        
        hgg_save_freq_dict = {'sawyer_door' : 2000,                                
                                'tabletop_manipulation' : 1000,
                                'fetch_reach_ergodic' : 1000,
                                'fetch_push_ergodic' : 1000,
                                'fetch_pickandplace_ergodic' : 1000,
                                'point_umaze' : 1000,                                
                                }
        num_K_dict = {'sawyer_door' : 2,                                
                    'tabletop_manipulation' : 5,
                    'fetch_reach_ergodic' : 50,
                    'fetch_push_ergodic' : 50,
                    'fetch_pickandplace_ergodic' : 50,
                    'point_umaze' : 50,                                
                    }
        
        cfg.max_episode_timesteps = max_episode_timesteps_dict[cfg.env]
        cfg.num_train_steps = num_train_steps_dict[cfg.env]  
        cfg.hgg_save_freq = hgg_save_freq_dict[cfg.env]
        cfg.num_K = num_K_dict[cfg.env]
            
        
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             action_repeat=cfg.action_repeat,
                             agent='ibc')

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        
        
        if cfg.env in ['tabletop_manipulation', 'sawyer_door', 'fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', 'point_umaze']:
            if cfg.env in ['tabletop_manipulation', 'sawyer_door']:
                import earl_benchmark
                env_loader = earl_benchmark.EARLEnvs(cfg.env, reward_type='sparse')
                env, eval_env = env_loader.get_envs()
                earl_env = True                
                
                if cfg.sparse_reward_type == 'negative': 
                    reward_offset = -1.0 # sparse reward [-1, 0] 
                elif cfg.sparse_reward_type == 'positive':
                    reward_offset = 0.0
                        
            elif cfg.env in ['fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', 'point_umaze']:
                import env_loader
                if cfg.full_state_goal:
                    assert cfg.env not in ['fetch_reach_ergodic', 'point_umaze']
                
                if cfg.xml_path in ['None', 'none', None]:
                    cfg.xml_path = None
                
                if 'fetch' in cfg.env:
                    loader = env_loader.GymEnvs(cfg.env+'2', reward_type="sparse", full_state_goal=cfg.full_state_goal, xml_path=cfg.xml_path)
                else:
                    loader = env_loader.GymEnvs(cfg.env, reward_type="sparse", full_state_goal=cfg.full_state_goal, xml_path=cfg.xml_path)

                env, eval_env = loader.get_envs()
                earl_env = True                
                if cfg.sparse_reward_type == 'negative':
                    reward_offset = 0.0 # sparse reward [-1, 0] 
                elif cfg.sparse_reward_type == 'positive': 
                    reward_offset = 1.0
                    

            if cfg.use_curriculum:
                '''
                NOTE : hgg env should be used only for obatining initial & final goal.
                Do not use for rollout as its (train env) horizon is long!
                '''
                if cfg.env in ['tabletop_manipulation', 'sawyer_door']:
                    hgg_env_loader = earl_benchmark.EARLEnvs(cfg.env, reward_type='sparse')
                    hgg_env, hgg_eval_env = hgg_env_loader.get_envs()                
                elif cfg.env in ['fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', 'point_umaze']:
                    if cfg.full_state_goal:
                        assert cfg.env not in ['fetch_reach_ergodic', 'point_umaze']
                    if 'fetch' in cfg.env:
                        hgg_env_loader = env_loader.GymEnvs(cfg.env+'2', reward_type="sparse", full_state_goal=cfg.full_state_goal, xml_path=cfg.xml_path)
                    else:
                        hgg_env_loader = env_loader.GymEnvs(cfg.env, reward_type="sparse", full_state_goal=cfg.full_state_goal, xml_path=cfg.xml_path)
                    hgg_env, hgg_eval_env = hgg_env_loader.get_envs()


            if cfg.no_backward_if_forward_succeed: 
                if cfg.env in ['tabletop_manipulation', 'sawyer_door', 'point_umaze']:                    
                    if cfg.env in ['sawyer_door']: 
                        assert cfg.num_K==2, 'testing init and final goal'
                        init_g = hgg_eval_env.initial_states[0,:].copy()
                        final_g = hgg_eval_env.goal_states[0,:].copy()
                        custom_task_goal = np.stack([init_g, final_g], axis=0) # [2, dim]

                    elif cfg.env=='tabletop_manipulation':
                        initial_states = env.initial_state.copy()[None, :]
                        goal_states = env.goal_states.copy()
                        custom_task_goal = np.concatenate([initial_states, goal_states], axis =0)
                    elif cfg.env=='point_umaze':
                        if not cfg.load_pretrained:
                            state_1 = np.linspace([0,0], [8,0], 10)
                            state_2 = np.linspace([8,0], [8,8], 10)
                            state_3 = np.linspace([8,8], [0,8], 10)
                            custom_task_goal = np.concatenate([state_1, state_2, state_3], axis =0)
                            custom_task_goal += np.random.uniform(-1,1, size=custom_task_goal.shape)*0.1
                        else:
                            custom_task_goal = np.array([[0.0, 8.0]])

                    
                    env.set_custom_task_goal(custom_task_goal)
                    if cfg.use_curriculum:
                        hgg_env.set_custom_task_goal(custom_task_goal)
                    # print(f'set custom task goal for env : {cfg.env}')

            
            sawyer_velocity_info = None
            if cfg.env=='sawyer_door':
                env.set_velocity_info(sawyer_velocity_info)
                eval_env.set_velocity_info(sawyer_velocity_info)
                if cfg.use_curriculum:
                    hgg_env.set_velocity_info(sawyer_velocity_info)
                    hgg_eval_env.set_velocity_info(sawyer_velocity_info)
                    
            if cfg.backward_proprioceptive_only:
                from env_utils import RewardChangeWrapperEnv
                env = RewardChangeWrapperEnv(env, env_name = cfg.env)
                eval_env = RewardChangeWrapperEnv(eval_env, env_name = cfg.env)
                if cfg.use_curriculum:
                    hgg_env = RewardChangeWrapperEnv(hgg_env, env_name = cfg.env)
                    hgg_eval_env = RewardChangeWrapperEnv(hgg_eval_env, env_name = cfg.env)

            from env_utils import RewardOffsetWrapper
            env = RewardOffsetWrapper(env, reward_offset=reward_offset)
            eval_env = RewardOffsetWrapper(eval_env, reward_offset=reward_offset)
            if cfg.use_curriculum:
                hgg_env = RewardOffsetWrapper(hgg_env, reward_offset=reward_offset)
                hgg_eval_env = RewardOffsetWrapper(hgg_eval_env, reward_offset=reward_offset)

                         
            from env_utils import NonEpisodicWrapper, DoneOnSuccessWrapper, HERGoalEnvWrapper

            if cfg.done_on_success:
                env = DoneOnSuccessWrapper(env, reward_offset=0.0, earl_env = earl_env)
                eval_env = DoneOnSuccessWrapper(eval_env, reward_offset=0.0, earl_env = earl_env)
                if cfg.use_curriculum:
                    hgg_env = DoneOnSuccessWrapper(hgg_env, reward_offset=0.0, earl_env = earl_env)
                    hgg_eval_env = DoneOnSuccessWrapper(hgg_eval_env, reward_offset=0.0, earl_env = earl_env)
            
            if cfg.env in ['fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', 'point_umaze']:
                env = HERGoalEnvWrapper(env, env_name= cfg.env)
                eval_env = HERGoalEnvWrapper(eval_env, env_name= cfg.env)
                if cfg.use_curriculum:
                    hgg_env = HERGoalEnvWrapper(hgg_env, env_name= cfg.env)
                    hgg_eval_env = HERGoalEnvWrapper(hgg_eval_env, env_name= cfg.env)

            env = NonEpisodicWrapper(env, cfg.env, forward_env_obs_type=cfg.forward_env_obs_type, backward_env_obs_type=cfg.backward_env_obs_type)                                    
            eval_env = NonEpisodicWrapper(eval_env, cfg.env, forward_env_obs_type=cfg.forward_env_obs_type, backward_env_obs_type=cfg.backward_env_obs_type)                    
            if cfg.use_curriculum:
                hgg_env = NonEpisodicWrapper(hgg_env, cfg.env, forward_env_obs_type=cfg.forward_env_obs_type, backward_env_obs_type=cfg.backward_env_obs_type)                    
                hgg_eval_env = NonEpisodicWrapper(hgg_eval_env, cfg.env, forward_env_obs_type=cfg.forward_env_obs_type, backward_env_obs_type=cfg.backward_env_obs_type)                    


            from env_utils import StateWrapper
            self.env = StateWrapper(env)
            self.eval_env = StateWrapper(eval_env)
            if cfg.use_curriculum:
                self.hgg_env = StateWrapper(hgg_env)
                self.hgg_eval_env = StateWrapper(hgg_eval_env)
            
            forward_obs_spec = self.env.observation_spec('forward')
            backward_obs_spec = self.env.observation_spec('backward')
            
            action_spec = self.env.action_spec()
        
        
        
        
        cfg.agent.action_shape = action_spec.shape
        cfg.forward_agent.action_shape = action_spec.shape

        
        cfg.agent.action_range = [
            float(action_spec.low.min()),
            float(action_spec.high.max())
        ]
        
        cfg.forward_agent.action_range = [
            float(action_spec.low.min()),
            float(action_spec.high.max())
        ]
        
        self.max_episode_timesteps = cfg.max_episode_timesteps


        cfg.state_dim = self.env.obs_dim
        if cfg.env in ['fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', 'point_umaze']:
            cfg.state_goal_dim = self.env.obs_dim+self.env.goal_dim*2 # obs, ag, dg
        else:
            cfg.state_goal_dim = self.env.obs_dim+self.env.goal_dim # obs(include ag), dg
        
        
        if cfg.traj_length in ['none', None, 'None']:
            cfg.traj_length = None

        cfg.forward_agent.obs_shape = forward_obs_spec.shape

        self.forward_agent = hydra.utils.instantiate(cfg.forward_agent)
        self.backward_agent = None
        
        cfg.agent.obs_shape = backward_obs_spec.shape
        
        self.backward_agent = hydra.utils.instantiate(cfg.agent)
    

        self.forward_buffer = ReplayBuffer(forward_obs_spec.shape, action_spec.shape,
                                        cfg.replay_buffer_capacity,
                                        self.device, traj_length=cfg.traj_length, env_name=cfg.env,
                                        )
        
        from env_utils import WraptoGoalEnv
        self.goal_env = goal_env = WraptoGoalEnv(self.env, env_name= cfg.env, sparse_reward_type = cfg.sparse_reward_type)
        
        if self.cfg.use_curriculum:
            self.hgg_goal_env = hgg_goal_env = WraptoGoalEnv(self.hgg_env, env_name= cfg.env, sparse_reward_type = cfg.sparse_reward_type)
            self.hgg_goal_eval_env = hgg_goal_eval_env = WraptoGoalEnv(self.hgg_eval_env, env_name= cfg.env, sparse_reward_type = cfg.sparse_reward_type)
        

        if cfg.use_forward_her:
            assert cfg.forward_env_obs_type=='state_goal'
            n_sampled_goal = 4                
            self.forward_buffer = HindsightExperienceReplayWrapperVer2(self.forward_buffer, 
                                                                    n_sampled_goal=n_sampled_goal,                                                                     
                                                                    wrapped_env=goal_env,
                                                                    env_name = cfg.env,
                                                                    )
            
            
        
        self.backward_buffer = ReplayBuffer(backward_obs_spec.shape, action_spec.shape,
                                        cfg.replay_buffer_capacity,
                                        self.device, traj_length=cfg.traj_length, env_name=cfg.env,
                                        )
            
        
        if cfg.use_backward_her:
            assert cfg.backward_env_obs_type=='state_goal'
            self.backward_buffer = HindsightExperienceReplayWrapperVer2(self.backward_buffer, 
                                                                        n_sampled_goal=n_sampled_goal,                                                                     
                                                                        wrapped_env=goal_env,
                                                                        env_name = cfg.env,
                                                                        )
        
        if cfg.use_curriculum:
            from hgg.hgg import TrajectoryPool, MatchSampler
            if cfg.forward_curriculum:
                self.forward_curriculum_achieved_trajectory_pool = TrajectoryPool(**cfg.hgg_kwargs.trajectory_pool_kwargs)
                self.forward_curriculum_sampler = MatchSampler(goal_env=self.hgg_goal_env, goal_eval_env = self.hgg_goal_eval_env, env_name=cfg.env, achieved_trajectory_pool = self.forward_curriculum_achieved_trajectory_pool, 
                                                agent_type='forward', **cfg.hgg_kwargs.match_sampler_kwargs)                
                self.forward_curriculum_sampler.set_networks(critic=self.forward_agent.critic, policy = self.forward_agent.actor)
            
            
            if cfg.backward_curriculum:
                self.backward_curriculum_achieved_trajectory_pool = TrajectoryPool(**cfg.hgg_kwargs.trajectory_pool_kwargs)
                self.backward_curriculum_sampler = MatchSampler(goal_env=self.hgg_goal_env,  goal_eval_env = self.hgg_goal_eval_env, env_name=cfg.env, achieved_trajectory_pool = self.backward_curriculum_achieved_trajectory_pool,
                                                agent_type='backward', **cfg.hgg_kwargs.match_sampler_kwargs)                
                self.backward_curriculum_sampler.set_networks(critic=self.backward_agent.critic, policy = self.backward_agent.actor)
            else:
                pass
                


        
        from replay_buffer import ForwardBackwardReplayBufferWrapper
        self.wrapped_replay_buffer = ForwardBackwardReplayBufferWrapper(self.forward_buffer, self.backward_buffer)
        self.non_episodic_video_recorder = VideoRecorder(self.work_dir+'/non_episodic' if cfg.save_video else None, dmc_env=False, env_name=cfg.env)
        

        self.eval_video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, dmc_env=False, env_name=cfg.env)
        self.step = 0
        self.forward_step = 0
        self.backward_step = 0

    def get_agent(self, option = 'forward'):
        if option=='forward':        
            return self.forward_agent
        elif option=='backward':
            return self.backward_agent
    

    def get_buffer(self, option = 'forward'):
        self.wrapped_replay_buffer.set_option(option)
        if option=='forward':
            if self.cfg.use_forward_her:                
                self.forward_buffer.replay_buffer.sample_type = None
            else:
                self.forward_buffer.sample_type = None
            return self.forward_buffer
        elif option=='backward':
            return self.backward_buffer
        
    def get_hgg_sampler(self, option='forward'):
        if option=='forward':
            return self.forward_curriculum_sampler
        elif option=='backward':
            return self.backward_curriculum_sampler

    def get_hgg_achieved_trajectory_pool(self, option='forward'):
        if option=='forward':
            return self.forward_curriculum_achieved_trajectory_pool
        elif option=='backward':
            return self.backward_curriculum_achieved_trajectory_pool

    def evaluate(self):
        avg_episode_reward = 0
        avg_episode_success_rate = 0
        if self.cfg.backward_proprioceptive_only:
            self.eval_env.set_proprioceptive_only(False)
            assert not self.eval_env.proprioceptive_only
            

        for episode in range(self.cfg.num_eval_episodes):
            observes = []
            
            obs = self.eval_env.reset()
            observes.append(obs)
            self.eval_video_recorder.init(enabled=(episode == 0))
            episode_reward = 0
            episode_success = 0
            episode_step = 0
            done = False
            
            agent = self.get_agent(option='forward')
            
            while not done:
                with utils.eval_mode(agent):                                        
                    action = agent.act(obs, goal_env=self.goal_env,  sample=False)
                next_obs, reward, done, info = self.eval_env.step(action)
                self.eval_video_recorder.record(self.eval_env)
                episode_reward += reward
                episode_step += 1
                obs = next_obs
                observes.append(obs)

            
            if self.eval_env.is_successful(obs):
                avg_episode_success_rate+=1.0
            
            avg_episode_reward += episode_reward
            self.eval_video_recorder.save(f'{self.step}.mp4')
        avg_episode_reward /= self.cfg.num_eval_episodes
        avg_episode_success_rate = avg_episode_success_rate/self.cfg.num_eval_episodes

        self.logger.log('eval/episode_reward', avg_episode_reward, self.step)
        self.logger.log('eval/episode_success_rate', avg_episode_success_rate, self.step)
        self.logger.dump(self.step, ty='eval')

    def run(self):
        self._run_non_episodic()
            
    

    def _run_non_episodic(self):        
        episode, episode_reward, episode_step = 0, 0, 0
        episode_observes =[]
        episode_rewards = []
        recent_non_episodic_episode_reward = Queue(50)
        if self.cfg.use_curriculum:
            recent_sampled_forward_goals = Queue(self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes)
            recent_sampled_backward_goals = Queue(self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes)
            

        recent_non_episodic_10k_steps = Queue(10000)
        recent_non_episodic_50k_steps = Queue(50000)
        recent_non_episodic_100k_steps = Queue(100000)
        
        recent_non_episodic_1k_episodes = Queue(1000)
        recent_non_episodic_100_episodes = Queue(100)
        

        forward_episode, backward_episode = 0,0
        
        episode += 1
        forward_episode +=1
        
        start_time = time.time()
        done = True        
        option = 'forward'
        init_state = None

        
        # pre-collect initial states
        initial_states = []
        full_initial_states = []
        if self.cfg.backward_proprioceptive_only: # to make achieved_goal in full_initial_states as gripper position
            self.env.set_proprioceptive_only(True)
        for i in range(1000):
            obs = self.env.reset()
            initial_states.append(obs[:self.env.obs_dim])
            full_initial_states.append(obs)
        initial_states = np.stack(initial_states, axis =0)
        full_initial_states = np.stack(full_initial_states, axis =0)
        if self.cfg.backward_proprioceptive_only:
            self.env.set_proprioceptive_only(False)

                
        if self.cfg.use_curriculum:
            temp_obs = self.hgg_goal_eval_env.reset()
            recent_sampled_forward_goals.put(self.hgg_goal_env.convert_obs_to_dict(temp_obs)['achieved_goal'].copy())
        
        
        if self.cfg.env in ['fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', 'point_umaze']:
            K = self.cfg.num_K
            assert K > 0
            predetermined_initial_goals = []
            predetermined_desired_goals = []
            for i in range(K):
                temp_obs = self.hgg_goal_env.convert_obs_to_dict(self.hgg_env.reset())
                goal_a = temp_obs['observation'].copy()
                if self.cfg.backward_proprioceptive_only:
                    goal_a = add_noise_to_goal(goal_a, self.cfg.env)
                goal_d = temp_obs['desired_goal'].copy()
                predetermined_initial_goals.append(goal_a.copy())
                predetermined_desired_goals.append(goal_d.copy())

        while self.step <= self.cfg.num_train_steps:            
            if done: # only done when training horizon is over                                                            
                self.env.set_option('forward', init_state=None)
                
                if self.cfg.backward_proprioceptive_only:
                    self.env.set_proprioceptive_only(False)

                obs = self.env.reset()
                print('done = True at step : ', self.step)
                if option=='backward':
                    forward_episode +=1
                    
                option = 'forward'
                episode_reward = 0
                episode_step = 0
                episode_observes = []
                episode_rewards = []
                
            elif self.step > 0:
                
                if last_timestep:
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()
                    if recent_non_episodic_episode_reward.full():
                        recent_non_episodic_episode_reward.get()
                    recent_non_episodic_episode_reward.put(episode_reward)
                    self.logger.log('train/recent_non_episodic_episode_reward', np.array(recent_non_episodic_episode_reward.queue).mean(), self.step)
                    
                    
                    if self.cfg.use_curriculum :
                        if self.cfg.env in ['tabletop_manipulation', 'sawyer_door', 'point_umaze']: # pure_obs==ag                            
                            assert self.env.custom_task_goal is not None      

                        if option=='forward' and self.cfg.forward_curriculum and (forward_episode % self.cfg.hgg_kwargs.hgg_sampler_update_frequency ==0 or forward_episode==5):
                            initial_goals = []
                            desired_goals = []
                            
                            if self.cfg.env in ['tabletop_manipulation', 'sawyer_door']:
                                # This process is just copying firstly assigned custom task goal.
                                K = self.cfg.num_K
                                assert K > 0 and K == self.env.custom_task_goal.shape[0]
                                
                                if self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes==K:                                        
                                    for i in range(self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes):
                                        temp_obs = self.hgg_goal_env.convert_obs_to_dict(self.hgg_env.reset())
                                        goal_a = temp_obs['achieved_goal'].copy()
                                        if self.cfg.backward_proprioceptive_only:
                                            goal_a = add_noise_to_goal(goal_a, self.cfg.env)
                                        goal_d = self.env.custom_task_goal[i].copy()
                                        initial_goals.append(goal_a.copy())
                                        desired_goals.append(goal_d.copy())
                                    
                                elif self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes > K:

                                    for i in range(self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes):
                                        temp_obs = self.hgg_goal_env.convert_obs_to_dict(self.hgg_env.reset())
                                        goal_a = temp_obs['achieved_goal'].copy()
                                        if self.cfg.backward_proprioceptive_only:
                                            goal_a = add_noise_to_goal(goal_a, self.cfg.env)
                                        temp_idx = int(i%K)
                                        goal_d = self.env.custom_task_goal[temp_idx].copy()
                                        initial_goals.append(goal_a.copy())
                                        desired_goals.append(goal_d.copy())
                                    
                                else:
                                    raise NotImplementedError
                                
                            
                            elif self.cfg.env in ['fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', 'point_umaze']:
                                initial_goals = copy.deepcopy(predetermined_initial_goals)
                                desired_goals = copy.deepcopy(predetermined_desired_goals)
                                   
                            
                            hgg_sampler = self.get_hgg_sampler(option=option)
                            hgg_sampler.update(initial_goals, desired_goals)
                            
                            
                    # non episodic
                    if option == 'forward':
                        self.logger.log('train/forward_episode_reward', episode_reward, self.step)
                        self.logger.log('train/forward_episode', forward_episode, self.step)
                        
                        forward_episode +=1
                        
                        # if forward was success keep option as forward
                        if (self.cfg.no_backward_if_forward_succeed and self.eval_env.is_successful(obs)):
                            option = 'forward'
                            
                            self.env.set_proprioceptive_only(False)
                            if self.env.forward_env_obs_type=='state_goal':                                                                
                                if self.cfg.forward_curriculum:
                                    hgg_sampler = self.get_hgg_sampler(option='forward')
                                    n_iter = 0
                                    
                                    while True:
                                        forward_goal = hgg_sampler.sample(np.random.randint(len(hgg_sampler.pool)), backward_proprioceptive=False).copy()
                                        # exclude already success goal
                                        obs_for_success_check = obs.copy()
                                        obs_for_success_check = self.env.replace_goal_in_obs(obs_for_success_check, forward_goal)
                                        
                                        if not self.env.is_successful(obs_for_success_check, proprioceptive_only=False):
                                            break
                                        n_iter +=1
                                        if n_iter==2:
                                            self.env.reset_goal(add_noise = self.cfg.add_noise_to_forward_goal)
                                            forward_goal = self.env.goal.copy().astype(np.float32)
                                            break
                                    
                                    self.env.reset_goal(goal=forward_goal, add_noise = self.cfg.add_noise_to_forward_goal) # random target goal for every episode
                                    forward_goal = self.env.goal.copy().astype(np.float32)
                                    if recent_sampled_forward_goals.full():
                                        recent_sampled_forward_goals.get()
                                    recent_sampled_forward_goals.put(forward_goal)
                                
                                else:
                                    prev_goal = self.goal_env.convert_obs_to_dict(obs)['desired_goal']                                    
                                    
                                    same_as_before = True
                                    while same_as_before:                                        
                                        self.env.reset_goal(add_noise = self.cfg.add_noise_to_forward_goal) # random target goal for every episode
                                        temp_goal = self.env.goal.copy().astype(np.float32)
                                        same_as_before = np.linalg.norm(prev_goal-temp_goal, axis =-1) < 0.001
                                    
                                    
                                forward_goal = self.env.goal.copy().astype(np.float32)                                   
                                obs_info = {'goal' : forward_goal}
                            else:
                                obs_info = None                        
                            
                            if self.env.backward_env_obs_type=='state_goal':
                                obs = self.env.replace_goal_in_obs(obs, forward_goal)
                            else:
                                raise NotImplementedError                                
                            # print('option is changed from forward to forward. goal in obs : {}'.format(self.env.get_pure_goal_from_obs(obs).squeeze()))
                            

                        else: # change from forward to backward 
                            option = 'backward'                            
                            if self.cfg.backward_proprioceptive_only:
                                self.env.set_proprioceptive_only(True)

                            if self.cfg.forward_curriculum and self.cfg.backward_proprioceptive_only : # consider backward curriculum goal for backward reaching
                                hgg_sampler = self.get_hgg_sampler(option='forward')
                                n_iter = 0
                                while True:
                                    backward_goal = hgg_sampler.sample(np.random.randint(len(hgg_sampler.pool)), backward_proprioceptive=True).copy()
                                    # exclude already success goal
                                    obs_for_success_check = obs.copy()
                                    obs_for_success_check = self.env.replace_goal_in_obs(obs_for_success_check, backward_goal)
                                    
                                    if not self.env.is_successful(obs_for_success_check, proprioceptive_only=True):
                                        break
                                    n_iter +=1
                                    if n_iter==2:
                                        break

                                if recent_sampled_backward_goals.full():
                                    recent_sampled_backward_goals.get()
                                recent_sampled_backward_goals.put(backward_goal)

                            elif self.cfg.use_curriculum and self.cfg.backward_curriculum : # consider backward curriculum goal (previous)
                                raise NotImplementedError

                            else: # just utilize initial state
                                if self.cfg.env in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:        
                                    if self.env.full_state_goal:
                                        assert full_initial_states.shape[-1]==37
                                        backward_goal = full_initial_states[np.random.randint(0, full_initial_states.shape[0]), -12:-6]
                                    else:
                                        assert full_initial_states.shape[-1]==31  
                                        backward_goal = full_initial_states[np.random.randint(0, full_initial_states.shape[0]), -6:-3]
                                elif self.cfg.env in ['fetch_reach_ergodic']:
                                    assert full_initial_states.shape[-1]==16
                                    backward_goal = full_initial_states[np.random.randint(0, full_initial_states.shape[0]), -6:-3]
                                elif self.cfg.env in ['point_umaze']:  
                                    assert full_initial_states.shape[-1]==11
                                    backward_goal = full_initial_states[np.random.randint(0, full_initial_states.shape[0]), -4:-2]                                    
                                elif self.cfg.env in ['sawyer_door']:                                    
                                    backward_goal = initial_states[np.random.randint(0, initial_states.shape[0]), :7]
                                else:
                                    backward_goal = initial_states[np.random.randint(0, initial_states.shape[0])]
                            
                                if self.cfg.backward_proprioceptive_only:
                                    backward_goal = add_noise_to_goal(backward_goal, self.cfg.env)

                            self.env.reset_goal(goal = backward_goal) # random target goal for every episode
                            goal = self.env.goal.copy().astype(np.float32)                                                               
                            assert (backward_goal==goal).all()
                            obs = self.env.replace_goal_in_obs(obs, goal)                            
                            obs = self.env.convert_obs(obs, option)

                            # print('option is changed from forward to backward. goal in obs : {}'.format(self.env.get_pure_goal_from_obs(obs).squeeze()))

                    elif option == 'backward':  # change from backward to forward 
                        self.logger.log('train/backward_episode_reward', episode_reward, self.step)
                        self.logger.log('train/backward_episode', backward_episode, self.step)
                                                
                        backward_episode +=1
                        option = 'forward'

                        if self.cfg.backward_proprioceptive_only:
                            self.env.set_proprioceptive_only(False)

                        if self.env.forward_env_obs_type=='state_goal':                            
                            if self.cfg.forward_curriculum and self.cfg.backward_proprioceptive_only:
                                hgg_sampler = self.get_hgg_sampler(option='forward')
                                n_iter = 0
                                while True:
                                    forward_goal = hgg_sampler.sample(np.random.randint(len(hgg_sampler.pool))).copy()
                                    # exclude already success goal
                                    obs_for_success_check = obs.copy()
                                    obs_for_success_check = self.env.replace_goal_in_obs(obs_for_success_check, forward_goal)
                                    
                                    if not self.env.is_successful(obs_for_success_check, proprioceptive_only=False):
                                        break
                                    n_iter +=1
                                    if n_iter==2:
                                        # sample task goal from env
                                        self.env.reset_goal(add_noise = self.cfg.add_noise_to_forward_goal)
                                        forward_goal = self.env.goal.copy().astype(np.float32)
                                        break

                                if recent_sampled_forward_goals.full():
                                    recent_sampled_forward_goals.get()
                                recent_sampled_forward_goals.put(forward_goal)
                                
                                self.env.reset_goal(goal = forward_goal, add_noise = self.cfg.add_noise_to_forward_goal)
                                if not self.cfg.add_noise_to_forward_goal:
                                    assert (self.env.goal.copy()==forward_goal).all()
                                
                            elif self.cfg.forward_curriculum : # consider backward curriculum goal (previous)
                                raise NotImplementedError

                            else:
                                self.env.reset_goal(add_noise = self.cfg.add_noise_to_forward_goal) # random target goal for every episode
                                
                            forward_goal = self.env.goal.copy().astype(np.float32)                                   
                            obs_info = {'goal' : forward_goal}
                        else:
                            obs_info = None                        

                        if self.env.backward_env_obs_type=='state_goal':
                            obs = self.env.replace_goal_in_obs(obs, forward_goal)
                        else:
                            obs = self.env.convert_obs(obs, option, obs_info)
                        # print('option is changed from backward to forward. goal in obs : {}'.format(self.env.get_pure_goal_from_obs(obs).squeeze()))

                    episode_reward = 0
                    episode_step = 0
                    episode_observes = []
                    episode_rewards = []
                    episode += 1

                    self.env.set_option(option=option, init_state=init_state)
                    

            agent = self.get_agent(option=option)
            replay_buffer = self.get_buffer(option=option)
            
            # evaluate agent periodically
            if self.step % self.cfg.eval_frequency == 0:
                print('eval started...')
                self.logger.log('eval/episode', episode - 1, self.step)
                self.evaluate()

            # save agent periodically
            if self.cfg.save_model and self.step % self.cfg.save_frequency == 0:
                utils.save(
                    self.forward_agent,
                    os.path.join(self.model_dir, f'forward_agent_{self.step}.pt'))
                utils.save(
                    self.backward_agent,
                    os.path.join(self.model_dir, f'backward_agent_{self.step}.pt'))
            
            if self.cfg.save_buffer and self.step % self.cfg.save_frequency == 0:
                utils.save(self.forward_buffer.sample_all_data(), os.path.join(self.forward_buffer_dir, f'forward_buffer.pt'))
                utils.save(self.backward_buffer.sample_all_data(), os.path.join(self.backward_buffer_dir, f'backward_buffer.pt'))
                

            if self.step % self.cfg.non_episodic_video_save_frequency == 0  or self.step in [4000, 8000, 12000, 16000]:
                self.non_episodic_video_recorder.init(enabled=True)
                non_episodic_obses = []
                non_episodic_actions = []
                non_episodic_rewards = []
                non_episodic_dones = []
                non_episodic_next_obses = []
                non_episodic_backward_indices_of_record = []
                non_episodic_idx_for_record = 0 
                

            # sample action for data collection
            if self.step < self.cfg.num_random_steps:
                spec = self.env.action_spec()                
                action = np.random.uniform(spec.low, spec.high,
                                        spec.shape)
                
            else: 
                with utils.eval_mode(agent):                        
                    action = agent.act(obs, goal_env=self.goal_env, sample=True)
                
            logging_dict = agent.update(self.wrapped_replay_buffer, self.step, self.goal_env)
            
            if self.forward_step % self.cfg.logging_frequency== 0:                
                if logging_dict is not None: # when step = 0                    
                    if option=='forward':
                        for key, val in logging_dict.items():
                            self.logger.log('train/forward/'+key, val, self.step)
                        # just for debug
                        self.logger.log('train/forward_step', self.forward_step, self.step)

            elif self.backward_step % self.cfg.logging_frequency== 0:                
                if logging_dict is not None: # when step = 0                    
                    if option=='backward':
                        for key, val in logging_dict.items():
                            self.logger.log('train/backward/'+key, val, self.step)
                        # just for debug
                        self.logger.log('train/backward_step', self.backward_step, self.step)


            if self.step>0 and (self.step % self.cfg.logging_frequency == 0):
                # just for logging
                self.logger.log('train/recent_non_episodic_10k_forward_ratio', np.array(recent_non_episodic_10k_steps.queue).mean(), self.step)
                self.logger.log('train/recent_non_episodic_50k_forward_ratio', np.array(recent_non_episodic_50k_steps.queue).mean(), self.step)
                self.logger.log('train/recent_non_episodic_100k_forward_ratio', np.array(recent_non_episodic_100k_steps.queue).mean(), self.step)
                self.logger.log('train/recent_non_episodic_10k_backward_ratio', 1-np.array(recent_non_episodic_10k_steps.queue).mean(), self.step)
                self.logger.log('train/recent_non_episodic_50k_backward_ratio', 1-np.array(recent_non_episodic_50k_steps.queue).mean(), self.step)
                self.logger.log('train/recent_non_episodic_100k_backward_ratio', 1-np.array(recent_non_episodic_100k_steps.queue).mean(), self.step)
                
                self.logger.log('train/recent_non_episodic_100_forward_episode_ratio', np.array(recent_non_episodic_100_episodes.queue).mean(), self.step)
                self.logger.log('train/recent_non_episodic_1k_forward_episode_ratio', np.array(recent_non_episodic_1k_episodes.queue).mean(), self.step)
                self.logger.log('train/recent_non_episodic_100_backward_episode_ratio', 1-np.array(recent_non_episodic_100_episodes.queue).mean(), self.step)
                self.logger.log('train/recent_non_episodic_1k_backward_episode_ratio', 1-np.array(recent_non_episodic_1k_episodes.queue).mean(), self.step)
            
            if self.step >=500 and ((self.step % self.cfg.hgg_save_freq == 0) or self.step in np.arange(0, 500*20+1, 500)):
                if self.cfg.forward_curriculum:
                    hgg_sampler = self.get_hgg_sampler(option='forward')
                    if hgg_sampler.total_cost is not None:
                        self.logger.log('train/forward_curriculum_total_cost_mean', hgg_sampler.total_cost.mean(), self.step)
                    if hgg_sampler.total_forward_cost is not None:
                        self.logger.log('train/forward_curriculum_total_forward_cost_mean', hgg_sampler.total_forward_cost.mean(), self.step)
                    if hgg_sampler.total_backward_cost is not None:
                        self.logger.log('train/forward_curriculum_total_backward_cost_mean', hgg_sampler.total_backward_cost.mean(), self.step)
                    
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    sampled_goals_for_vis = np.array(recent_sampled_forward_goals.queue)
                    if self.cfg.env in ['fetch_reach_ergodic', 'fetch_pickandplace_ergodic']:
                        pass
                    else:
                        # plot and save                                
                        fig = plt.figure()
                        sns.set_style("darkgrid")
                        ax1 = fig.add_subplot(1,1,1)
                        if self.cfg.env in ['point_umaze']:
                            ax1.scatter(sampled_goals_for_vis[:, 0], sampled_goals_for_vis[:, 1], c='red')
                            plt.xlim(-2,10)    
                            plt.ylim(-2,10)
                        elif self.cfg.env == 'tabletop_manipulation':
                            ax1.scatter(sampled_goals_for_vis[:, 2], sampled_goals_for_vis[:, 3], c='red')
                            ax1.scatter(sampled_goals_for_vis[:, 0], sampled_goals_for_vis[:, 1], c='blue')
                            plt.xlim(-2.8,2.8)    
                            plt.ylim(-2.8,2.8)
                        elif self.cfg.env == 'sawyer_door':
                            ax1.scatter(sampled_goals_for_vis[:, 4], sampled_goals_for_vis[:, 5], c='red')
                            plt.xlim(-0.2,0.4)    
                            plt.ylim(0.3,0.9)
                        elif self.cfg.env == 'fetch_push_ergodic':
                            ax1.scatter(sampled_goals_for_vis[:, 0], sampled_goals_for_vis[:, 1], c='red')
                            plt.xlim(0.8,1.8)    
                            plt.ylim(0.2,1.2)
                        else:
                            raise NotImplementedError
                        plt.savefig(self.non_episodic_video_recorder.save_dir+'/train_hgg_forward_goals_step_'+str(self.step)+'.jpg')
                        plt.close()
                    
                    with open(self.non_episodic_video_recorder.save_dir+'/train_hgg_forward_goals_step_'+str(self.step)+'.pkl', 'wb') as f:
                        pkl.dump(sampled_goals_for_vis, f)

                    if self.cfg.backward_proprioceptive_only:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        sampled_goals_for_vis = np.array(recent_sampled_backward_goals.queue)
                        if self.cfg.env in ['fetch_reach_ergodic', 'fetch_pickandplace_ergodic']:
                            pass
                        elif self.cfg.env in ['sawyer_door']:
                            pass
                        elif self.cfg.env == 'fetch_push_ergodic':
                            pass
                        else:
                            # plot and save    
                            fig = plt.figure()
                            sns.set_style("darkgrid")
                            ax1 = fig.add_subplot(1,1,1)
                            if self.cfg.env in ['point_umaze']:
                                ax1.scatter(sampled_goals_for_vis[:, 0], sampled_goals_for_vis[:, 1], c='blue')
                                plt.xlim(-2,10)    
                                plt.ylim(-2,10)
                            elif self.cfg.env == 'tabletop_manipulation':
                                ax1.scatter(sampled_goals_for_vis[:, 0], sampled_goals_for_vis[:, 1], c='blue')
                                plt.xlim(-2.8,2.8)
                                plt.ylim(-2.8,2.8)                                
                            else:
                                raise NotImplementedError
                            plt.savefig(self.non_episodic_video_recorder.save_dir+'/train_hgg_backward_goals_step_'+str(self.step)+'.jpg')
                            plt.close()
                        with open(self.non_episodic_video_recorder.save_dir+'/train_hgg_backward_goals_step_'+str(self.step)+'.pkl', 'wb') as f:
                            pkl.dump(sampled_goals_for_vis, f)




            next_obs, reward, done, info = self.env.step(action)
            
            episode_reward += reward
            
            last_timestep = True if (episode_step+1) % self.max_episode_timesteps == 0 or done else False
            
            if last_timestep:
                # NOTE: It is meaningless when consider_done_true_in_critic is False
                if self.cfg.done_on_success: # earl done or success or max episode length 
                    earl_done = info['earl_done'] # last_timestep not always means earl_done=True
                    if earl_done: # earl_done & (success or max episode timestep) simultaneously rarely happen -> ignore
                        done = False
                    else: # success or max episode length
                        # sparse reward
                        if done : # success 
                            pass
                        else: # max episode step
                            done = False 

                else: # earl done or max episode length
                    if done: # earl done = True regardless of steps (assume done from env is False)
                        earl_done=True                            
                    else: # max episode step
                        earl_done=False
                    done = False # done = False as done_on_success=False
                    
            episode_observes.append(obs)
            episode_rewards.append(reward)
            
            if self.cfg.use_forward_her or self.cfg.use_backward_her:
                if option=='forward' :                        
                    assert self.cfg.use_forward_her
                    replay_buffer.add(obs, action, reward, next_obs, done, last_timestep)
                        
                elif option=='backward' :
                    assert self.cfg.use_backward_her
                    replay_buffer.add(obs, action, reward, next_obs, done, last_timestep)

                        
            else: # no her
                # TODO: you should consider forward, backward                
                replay_buffer.add(obs, action, reward, next_obs, done)
                    
        
            if last_timestep:                    
                if self.cfg.use_curriculum:
                    if option=='forward' and (not self.cfg.forward_curriculum):
                        pass
                    elif option=='backward' and (not self.cfg.backward_curriculum) and (not self.cfg.backward_proprioceptive_only):
                        pass
                    else:
                        if option=='forward': # only insert traj in forward hgg pool
                            temp_episode_observes = copy.deepcopy(episode_observes)
                            temp_episode_ag = []
                            if self.cfg.env in ['tabletop_manipulation', 'sawyer_door']:
                                temp_episode_init = self.goal_env.convert_obs_to_dict(temp_episode_observes[0])['observation'] # NOTE : should it be [obs, ag] ?
                            elif self.cfg.env in ['fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', 'point_umaze']:
                                temp_episode_init_obs = self.goal_env.convert_obs_to_dict(temp_episode_observes[0])['observation']
                                temp_episode_init_ag = self.goal_env.convert_obs_to_dict(temp_episode_observes[0])['achieved_goal']
                                temp_episode_init = np.concatenate([temp_episode_init_obs, temp_episode_init_ag], axis =-1)
                            else:
                                raise NotImplementedError
                            
                            for k in range(len(temp_episode_observes)):
                                if self.cfg.env in ['tabletop_manipulation', 'sawyer_door']:
                                    temp_episode_ag.append(self.goal_env.convert_obs_to_dict(temp_episode_observes[k])['achieved_goal']) # pure_obs == ag
                                elif self.cfg.env in ['fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic', 'point_umaze']:                                
                                    temp_episode_ag.append(self.goal_env.convert_obs_to_dict(temp_episode_observes[k])['observation']) # pure_obs != ag
                                else:
                                    raise NotImplementedError

                            # NOTE : currently, add episodewise, thus list has only 1 element
                            achieved_trajectories = [np.array(temp_episode_ag)] # list of [ts, dim]
                            achieved_init_states = [temp_episode_init] # list of [ts(1), dim]

                            selection_trajectory_idx = {}
                            for i in range(len(achieved_trajectories)): # 1
                                # if there is a difference btw first and last timestep, then add it.
                                if self.cfg.backward_proprioceptive_only: 
                                    if self.cfg.env in ['sawyer_door', 'tabletop_manipulation', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
                                        if 'sawyer' in self.cfg.env: # consider gripper, object related states
                                            if goal_distance(achieved_trajectories[i][0][4:7], achieved_trajectories[i][-1][4:7])>0.01 or goal_distance(achieved_trajectories[i][0][:3], achieved_trajectories[i][-1][:3])>0.01:
                                                selection_trajectory_idx[i] = True
                                        elif self.cfg.env=='tabletop_manipulation': # only consider object related states
                                            if goal_distance(achieved_trajectories[i][0][2:4], achieved_trajectories[i][-1][2:4])>0.01 or goal_distance(achieved_trajectories[i][0][:2], achieved_trajectories[i][-1][:2])>0.01:
                                                selection_trajectory_idx[i] = True                                
                                        elif self.cfg.env in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
                                            if goal_distance(achieved_trajectories[i][0][3:6], achieved_trajectories[i][-1][3:6])>0.01 or goal_distance(achieved_trajectories[i][0][:3], achieved_trajectories[i][-1][:3])>0.01:
                                                selection_trajectory_idx[i] = True 
                                    
                                    
                                    elif self.cfg.env in ['fetch_reach_ergodic']:
                                        if goal_distance(achieved_trajectories[i][0][:3], achieved_trajectories[i][-1][:3])>0.01:
                                            selection_trajectory_idx[i] = True                                 
                                    elif self.cfg.env in ['point_umaze']:
                                        if goal_distance(achieved_trajectories[i][0][:2], achieved_trajectories[i][-1][:2])>0.1:
                                            selection_trajectory_idx[i] = True      
                                    else: # full state achieved_goal
                                        if goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01:
                                            selection_trajectory_idx[i] = True
                                else:
                                    if 'sawyer' in self.cfg.env: # only consider object related states
                                        if goal_distance(achieved_trajectories[i][0][4:7], achieved_trajectories[i][-1][4:7])>0.01:
                                            selection_trajectory_idx[i] = True
                                    elif self.cfg.env=='tabletop_manipulation': # only consider object related states
                                        if goal_distance(achieved_trajectories[i][0][2:4], achieved_trajectories[i][-1][2:4])>0.01:
                                            selection_trajectory_idx[i] = True                                
                                    elif self.cfg.env in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
                                        if goal_distance(achieved_trajectories[i][0][3:6], achieved_trajectories[i][-1][3:6])>0.01:
                                            selection_trajectory_idx[i] = True 
                                    elif self.cfg.env in ['fetch_reach_ergodic']:
                                        if goal_distance(achieved_trajectories[i][0][:3], achieved_trajectories[i][-1][:3])>0.01:
                                            selection_trajectory_idx[i] = True                                 
                                    elif self.cfg.env in ['point_umaze']:
                                        if goal_distance(achieved_trajectories[i][0][:2], achieved_trajectories[i][-1][:2])>0.1:
                                            selection_trajectory_idx[i] = True      
                                    else: # full state achieved_goal
                                        if goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01:
                                            selection_trajectory_idx[i] = True
                            if option=='backward' and self.cfg.backward_proprioceptive_only:
                                hgg_achieved_trajectory_pool = self.get_hgg_achieved_trajectory_pool(option='forward')
                            else:
                                hgg_achieved_trajectory_pool = self.get_hgg_achieved_trajectory_pool(option=option)

                            for idx in selection_trajectory_idx.keys():
                                hgg_achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())
                            
                        # visualize recent N sampled goals
                        if option=='forward':
                            recent_sampled_goals = recent_sampled_forward_goals
                        elif option=='backward':
                            recent_sampled_goals = recent_sampled_backward_goals
                        
                        # TODO : should consider how to visualize or whether to visualize.
                        if self.cfg.backward_proprioceptive_only:
                            if 'sawyer' in self.cfg.env :                                     
                                if option=='forward':
                                    self.env.set_tstar_states(hand_pos=None, obj_pos=np.array(recent_sampled_goals.queue)[:, 4:7]) # [n_queue, obj dim]
                                elif option=='backward':
                                    self.env.set_tstar_states(hand_pos=np.array(recent_sampled_goals.queue)[:, :3], obj_pos=None) # [n_queue, ee dim]

                            elif self.cfg.env=='tabletop_manipulation':                                                                        
                                if option=='forward':
                                    self.env.set_tstar_states(hand_pos=np.array(recent_sampled_goals.queue)[:, 0:2], obj_pos=np.array(recent_sampled_goals.queue)[:, 2:4])
                                elif option=='backward':                                    
                                    self.env.set_tstar_states(hand_pos=np.array(recent_sampled_goals.queue)[:, 0:2], obj_pos=None)
                            elif self.cfg.env in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
                                if option=='forward':
                                    self.env.set_tstar_states(hand_pos=None, obj_pos=np.array(recent_sampled_goals.queue)) # [n_queue, obj dim]
                                elif option=='backward':
                                    self.env.set_tstar_states(hand_pos=np.array(recent_sampled_goals.queue), obj_pos=None) # [n_queue, obj dim]
                            elif self.cfg.env =='point_umaze':
                                if option=='forward':
                                    self.env.set_tstar_states(hand_pos=None, obj_pos=np.array(recent_sampled_goals.queue)) # [n_queue, obj dim]
                                elif option=='backward':
                                    self.env.set_tstar_states(hand_pos=np.array(recent_sampled_goals.queue), obj_pos=None) # [n_queue, obj dim]
                            else:
                                pass
                                # raise NotImplementedError
                        else:
                            if 'sawyer' in self.cfg.env : 
                                sampled_goals_for_vis = np.array(recent_sampled_goals.queue)[:, 4:7] # [n_queue, obj dim]
                                if option=='forward':
                                    self.env.set_tstar_states(hand_pos=sampled_goals_for_vis, obj_pos=None)
                                elif option=='backward':
                                    self.env.set_tstar_states(hand_pos=None, obj_pos=sampled_goals_for_vis)
                            elif self.cfg.env=='tabletop_manipulation':
                                assert not (self.cfg.forward_curriculum and self.cfg.backward_curriculum), "currently, assume only one of forward or backward curriculum goal hgg"                                
                                sampled_goals_for_vis_hand = np.array(recent_sampled_goals.queue)[:, 0:2] # [n_queue, hand dim]
                                sampled_goals_for_vis_obj = np.array(recent_sampled_goals.queue)[:, 2:4] # [n_queue, hand dim]
                                if option=='forward':
                                    self.env.set_tstar_states(hand_pos=sampled_goals_for_vis_hand, obj_pos=sampled_goals_for_vis_obj)
                                elif option=='backward':                                    
                                    self.env.set_tstar_states(hand_pos=sampled_goals_for_vis_hand, obj_pos=sampled_goals_for_vis_obj)
                            else:
                                raise NotImplementedError
                        
                    
                

            if self.non_episodic_video_recorder.enabled:
                non_episodic_obses.append(obs)
                non_episodic_actions.append(action)
                non_episodic_rewards.append(reward)
                non_episodic_next_obses.append(next_obs)
                non_episodic_dones.append(done)
                
                if option=='backward':                    
                    non_episodic_backward_indices_of_record.append(non_episodic_idx_for_record)

                non_episodic_idx_for_record+=1


            self.non_episodic_video_recorder.record(self.env) # If enabled=False, do nothing
            if self.non_episodic_video_recorder.num_recorded_frames == self.cfg.num_non_episodic_record_frames:
                self.non_episodic_video_recorder.save(f'{self.step+1-self.cfg.num_non_episodic_record_frames}_{self.step+1}.mp4')                
                self.non_episodic_video_recorder.init(enabled=False)
                # print(f'non episodic video for {self.step+1-self.cfg.num_non_episodic_record_frames}_{self.step+1} steps is saved')
                
                # save data of non episodic video recording
                non_episodic_recording_data_dict = {'observation' : non_episodic_obses,#np.stack(non_episodic_obses, axis =0),
                                                    'action' : np.stack(non_episodic_actions, axis =0),
                                                    'done' : np.stack(non_episodic_dones, axis =0),    
                                                    'reward' : np.stack(non_episodic_rewards, axis =0),
                                                    'next_observation' : non_episodic_next_obses, #np.stack(non_episodic_next_obses, axis =0),
                                                    }
                
                
                path = os.path.join(self.non_episodic_video_recorder.save_dir, f'{self.step+1-self.cfg.num_non_episodic_record_frames}_{self.step+1}.pkl')
                with open(path, 'wb') as f:
                    pkl.dump(non_episodic_recording_data_dict, f)


            obs = next_obs
            if last_timestep:                                
                if earl_done:
                    done = True
                else:
                    done = False

                # just for logging
                if recent_non_episodic_100_episodes.full():
                    recent_non_episodic_100_episodes.get()
                if recent_non_episodic_1k_episodes.full():
                    recent_non_episodic_1k_episodes.get()

                if option=='forward':
                    recent_non_episodic_100_episodes.put(1)
                    recent_non_episodic_1k_episodes.put(1)
                elif option=='backward':
                    recent_non_episodic_100_episodes.put(0)
                    recent_non_episodic_1k_episodes.put(0)

            episode_step += 1
            self.step += 1
            
            # just for logging
            if recent_non_episodic_10k_steps.full():
                recent_non_episodic_10k_steps.get()
            if recent_non_episodic_50k_steps.full():
                recent_non_episodic_50k_steps.get()
            if recent_non_episodic_100k_steps.full():
                recent_non_episodic_100k_steps.get()

            if option=='forward':
                self.forward_step +=1                
                recent_non_episodic_10k_steps.put(1)
                recent_non_episodic_50k_steps.put(1)
                recent_non_episodic_100k_steps.put(1)
            elif option=='backward':
                self.backward_step +=1
                recent_non_episodic_10k_steps.put(0)
                recent_non_episodic_50k_steps.put(0)
                recent_non_episodic_100k_steps.put(0)
    
                


config_name = 'config_default.yaml'

@hydra.main(config_path='./config', config_name=config_name)
def main(cfg):
    import os
    os.environ['HYDRA_FULL_ERROR'] = str(1)
    from train import Workspace as W

    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
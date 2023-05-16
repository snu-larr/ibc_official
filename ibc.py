import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import copy
import math
import utils
import hydra
import time



def add_noise_to_goal(goal, env_name):
    # assume pure_obs    
    if 'sawyer' in env_name: #[ee(3), grip(1), obj(3)]
        assert goal.shape[-1]==7
        noise = np.random.uniform(-1,1, size=goal.shape)*0.05    
    elif env_name=='tabletop_manipulation': #[ee(2), obj(2), grip_state(2)]
        assert goal.shape[-1]==6
        noise = np.random.uniform(-1,1, size=goal.shape)*0.05
    elif 'fetch_reach_ergodic' in env_name: #[grip_pos(3), grip_state(2), grip_velp(3), gripper_vel(2)]
        assert goal.shape[-1]==3 or goal.shape[-1]==10 
        noise = np.random.uniform(-1,1, size=goal.shape)*0.05        
    elif env_name in ['fetch_pickandplace_ergodic', 'fetch_push_ergodic' ]: #[grip_pos(3), object_pos(3), ...]
        assert goal.shape[-1]==3 or goal.shape[-1]==6 or goal.shape[-1]==25
        noise = np.random.uniform(-1,1, size=goal.shape)*0.05    
    elif env_name=='point_umaze': # maybe [pos(3), vel(3), time(1)]
        assert goal.shape[-1]==2 or goal.shape[-1]==7
        noise = np.random.uniform(-1,1, size=goal.shape)*0.1    
    else:
        raise NotImplementedError
    noise_added_goal = (goal + noise).astype(np.float32)
    if env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
        if noise_added_goal.shape[-1]==6 or noise_added_goal.shape[-1]==25:
            noise_added_goal[..., 2] = np.clip(noise_added_goal[..., 2], 0.42, 1.0) # gripper_z
            if env_name=='fetch_push_ergodic':
                noise_added_goal[..., 5] = np.clip(noise_added_goal[..., 5], 0.42, 0.43) # object_z
            elif env_name=='fetch_pickandplace_ergodic':            
                noise_added_goal[..., 5] = np.clip(noise_added_goal[..., 5], 0.42, 1.0) # object_z                
            noise_added_goal[..., 3] = np.clip(noise_added_goal[..., 3], 1.19786948, 1.49786948) # object_x
            noise_added_goal[..., 4] = np.clip(noise_added_goal[..., 4], 0.59894948, 0.89894948) # object_y            
        elif noise_added_goal.shape[-1]==3:
            noise_added_goal[..., 2] = np.clip(noise_added_goal[..., 2], 0.42, np.inf) # gripper or object_z
            noise_added_goal[..., 0] = np.clip(noise_added_goal[..., 0], 1.19786948, 1.49786948) # gripper or object_x
            noise_added_goal[..., 1] = np.clip(noise_added_goal[..., 1], 0.59894948, 0.89894948) # gripper or object_y            
        else:
            raise NotImplementedError
    elif env_name =='tabletop_manipulation':
        noise_added_goal[..., :4] = np.clip(noise_added_goal[..., :4], -2.8, 2.8)
    elif env_name == 'sawyer_door':
        noise_added_goal[..., 6] = np.clip(noise_added_goal[..., 6], 0.1, 0.11) # door z position
    return noise_added_goal

def normalize_obs(obs, env_name, device=None): 
    
    if obs is None:
        return None
    if type(obs)==np.ndarray:
        obs = obs.copy()    
    elif type(obs)==torch.Tensor:
        obs = copy.deepcopy(obs)
    else:
        raise NotImplementedError

    if env_name in ['point_umaze']:
        # normalize to [-1,1]
        if env_name=='point_umaze':
            assert obs.shape[-1]==11
        
        if torch.is_tensor(obs):
            center, scale = torch.from_numpy(np.array([4.0, 4.0])).float().to(device), torch.from_numpy(np.array([6.0, 6.0])).float().to(device)
            obs[..., -4:] = (obs[..., -4:]-torch.tile(center, (2,)))/torch.tile(scale, (2,))
        else:       
            center, scale = np.array([4.0, 4.0]), np.array([6.0, 6.0])
            obs[..., -4:] = (obs[..., -4:]-np.tile(center, 2))/np.tile(scale,2)
        obs[..., :2] = (obs[..., :2]-center)/scale
        
    elif env_name in ['tabletop_manipulation', 'sawyer_door',  'fetch_reach_ergodic', 'fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
        raise NotImplementedError('normalization maybe not needed')
    else:
        raise NotImplementedError

    return obs



class IdentityEncoder(nn.Module):
    def __init__(self, repr_dim,  project_for_state_input = False):
        super().__init__()        
        # assert len(obs_shape) == 1
        
        self.repr_dim = repr_dim #obs_shape[-1]
        
        self.project_for_state_input = project_for_state_input
        if project_for_state_input:
            self.projector = nn.Linear(self.repr_dim, self.repr_dim)

    def encode(self, obs):
        return obs

    def forward(self, obs):
        h = self.encode(obs)
        if self.project_for_state_input:
            z = self.projector(h)
        else:
            z = h
            
        return z


class StateActor(nn.Module):
    def __init__(self, feature_dim, action_shape, hidden_dim,
                 hidden_depth, log_std_bounds,
                 fc_layer_norm_for_obs = False, repr_dim = None):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        # self.pre_fc = nn.Sequential(nn.Linear(repr_dim, feature_dim),
        #                             nn.LayerNorm(feature_dim))
        self.fc_layer_norm_for_obs = fc_layer_norm_for_obs
        if fc_layer_norm_for_obs: # repr_dim is obs dim            
            self.trunk = nn.Sequential(# convert image/state to a normalized vector 
                                        nn.Linear(repr_dim, feature_dim),
                                        nn.LayerNorm(feature_dim),
                                        nn.Tanh())
        else:
            feature_dim = repr_dim
        self.fc = utils.mlp(feature_dim, hidden_dim, 2 * action_shape[0],
                            hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, obs):
        if self.fc_layer_norm_for_obs:
            h = self.trunk(obs)
        else:            
            h = obs
        mu, log_std = self.fc(h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)
        return dist

class StateCritic(nn.Module):
    def __init__(self, feature_dim, action_shape, hidden_dim,
                 hidden_depth, 
                 fc_layer_norm_for_obs = False, repr_dim = None):
        super().__init__()

        # self.pre_fc = nn.Sequential(nn.Linear(repr_dim, feature_dim),
        #                             nn.LayerNorm(feature_dim))
        
        self.fc_layer_norm_for_obs = fc_layer_norm_for_obs
        
        
        if fc_layer_norm_for_obs: # repr_dim is obs dim            
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                        nn.LayerNorm(feature_dim),
                                        nn.Tanh())

            self.Q1 = utils.mlp(feature_dim + action_shape[0], hidden_dim, 1,
                                hidden_depth)
            self.Q2 = utils.mlp(feature_dim + action_shape[0], hidden_dim, 1,
                                hidden_depth)
        else:
            feature_dim = repr_dim
            self.Q1 = utils.mlp(feature_dim + action_shape[0], hidden_dim, 1,
                                hidden_depth)
            self.Q2 = utils.mlp(feature_dim + action_shape[0], hidden_dim, 1,
                                hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        # h = self.pre_fc(obs)
        
        if self.fc_layer_norm_for_obs:
            h = self.trunk(obs)
        else:
            h = obs
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2



class IBCAgent(object):
    def __init__(self, obs_shape, action_shape, action_range, device,
                 encoder_cfg, encoder_target_cfg, critic_cfg, critic_target_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency,
                 critic_target_tau, critic_target_update_frequency,
                 encoder_target_tau, encoder_update_frequency, batch_size,
                 num_seed_steps,
                 agent_type,
                 env_name = None,                 
                 inv_init= False, consider_done_true_in_critic = False,
                 env_obs_type = None,
                 adam_eps = 1e-8,                 
                 backward_proprioceptive_only = False,
                 normalize_rl_obs = False,
                 ):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_tau = critic_target_tau
        self.critic_target_update_frequency = critic_target_update_frequency
        self.encoder_target_tau = encoder_target_tau
        self.encoder_update_frequency = encoder_update_frequency
        self.batch_size = batch_size
        self.num_seed_steps = num_seed_steps
        self.lr = lr
        self.env_obs_type = env_obs_type
        self.custom_alpha_optimize = True
        self.normalize_rl_obs = normalize_rl_obs
        
        self.encoder = encoder_cfg.to(self.device)
        self.encoder_target = encoder_target_cfg.to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        
        critic_cfg.repr_dim = self.encoder.repr_dim
        self.critic = critic_cfg.to(self.device)
        self.critic_target = critic_target_cfg.to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())


        actor_cfg.repr_dim = self.encoder.repr_dim
        self.actor = actor_cfg.to(self.device)
        
        
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.env_name = env_name
        self.inv_init = inv_init
        self.alpha_lr = 1e-5
        self.agent_type = agent_type
        self.consider_done_true_in_critic = consider_done_true_in_critic
        self.adam_eps = adam_eps
        self.backward_proprioceptive_only = backward_proprioceptive_only
        
        if backward_proprioceptive_only:
            assert self.agent_type=='backward'

        self.is_first_actor_update = True
        
        # Changed target entropy from -dim(A) -> -dim(A)/2
        self.target_entropy = -action_shape[0] # /2.0 
        # optimizers
        self.init_optimizers(lr)

        self.train()
        self.critic_target.train()
        self.encoder_target.train()


    def init_optimizers(self, lr):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps = self.adam_eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr, eps = self.adam_eps)
        
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr, eps = self.adam_eps)
        
    
    def assign_modules_from(self, other):
        self.encoder = other.encoder
        self.encoder_target = other.encoder_target
        
        self.actor = other.actor
        # init opts
        self.init_optimizers(self.lr)

    def train(self, training=True):
        
        self.training = training    
        self.actor.train(training)
        self.critic.train(training)
        self.encoder.train(training)
        

    @property
    def alpha(self):        
        return self.log_alpha.exp()
        

    def act(self, obs, goal_env=None, sample=False, return_np = True):
        if self.normalize_rl_obs:
            obs = normalize_obs(obs, self.env_name, device=self.device)

        # obs = torch.FloatTensor(obs).to(self.device) # overuse cpu
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs).float().to(self.device)
        
        single_obs_input = True if obs.ndim==1 else False
        if single_obs_input:                
            obs = obs.unsqueeze(0)

        obs = self.encoder.encode(obs)
        
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)

        assert action.ndim == 2
        
        if single_obs_input:
            assert action.shape[0] == 1
            if return_np:
                return utils.to_np(action[0])
            else:
                return action[0]
        else: # batch obs input
            if return_np:
                return utils.to_np(action)
            else:
                return action

        # return utils.to_np(action[0])        
            
    def update_critic(self, obs, action, reward, next_obs, discount, done, step):
        
        with torch.no_grad():
            
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            
            assert len(next_action.shape)==2 and len(log_prob.shape)==2

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                target_Q2) - self.alpha.detach() * log_prob
            
            # target_Q = reward + (discount * target_V)
            if self.consider_done_true_in_critic:
                target_Q = reward + (discount * target_V)*(1-done)
            else:
                target_Q = reward + (discount * target_V)

        
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    

        return Q1, Q2, critic_loss

    def update_actor_and_alpha(self, obs, step):
        
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()


        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        
        # use custom alpha update             
        if self.custom_alpha_optimize:
            alpha_loss = (self.alpha*(-log_prob - self.target_entropy).detach()).mean() # just for logging
            alpha_loss_grad = (-log_prob - self.target_entropy).detach().mean()
            alpha = torch.clamp(self.alpha - self.alpha_lr*alpha_loss_grad , min=0.001, max = 0.5)
            self.log_alpha = alpha.log()

        else:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                        (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, log_prob
    
    

    def update(self, replay_buffer, step, goal_env = None):        
        return self._update(replay_buffer, step, goal_env)

    

    def _update(self, replay_buffer, step, goal_env = None):
        if (self.agent_type=='backward' and self.backward_proprioceptive_only):
            proprioceptive_only = True 
            assert replay_buffer.option=='backward'
        else:
            proprioceptive_only = False


        # if len(replay_buffer) < self.num_seed_steps:
        if step < self.num_seed_steps:
            return

        # just for sanity check
        if self.agent_type=='forward':
            assert replay_buffer.option=='forward'
        elif self.agent_type=='backward':
            assert replay_buffer.option=='backward'
            
        
        obs, action, extr_reward, next_obs, discount, dones = replay_buffer.sample(self.batch_size, self.discount, proprioceptive_only=proprioceptive_only) # Assume use HER
        
        reward = extr_reward
                
        if self.normalize_rl_obs:
            obs = normalize_obs(obs, self.env_name, device=self.device)
            next_obs = normalize_obs(next_obs, self.env_name, device=self.device)

        # decouple representation
        with torch.no_grad():
            obs = self.encoder.encode(obs)
            next_obs = self.encoder.encode(next_obs)
        
        Q1, Q2, critic_loss = self.update_critic(obs, action, reward, next_obs, discount, dones, step)

        if step % self.actor_update_frequency == 0 or self.is_first_actor_update:
            self.actor_loss, self.alpha_loss, self.actor_log_prob = self.update_actor_and_alpha(obs, step)
            self.is_first_actor_update = False

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                    self.critic_target_tau)
            
        # logging
        logging_dict = dict(q1=Q1.detach().cpu().numpy().mean(),
                            q2=Q2.detach().cpu().numpy().mean(),
                            critic_loss=critic_loss.detach().cpu().numpy(),
                            actor_loss = self.actor_loss.detach().cpu().numpy(),                            
                            batch_reward_mean = reward.detach().cpu().numpy().mean(),                            
                            )
        
        logging_dict.update(dict(
                                alpha_loss = self.alpha_loss.detach().cpu().numpy(),
                                bacth_actor_log_prob = self.actor_log_prob.detach().cpu().numpy().mean(),
                                # alpha = self.alpha.detach().cpu().numpy(),
                                entropy_diff = (-self.actor_log_prob-self.target_entropy).detach().cpu().numpy().mean(),
                                ))
        
        logging_dict.update(dict(alpha = self.alpha.detach().cpu().numpy()))


        return logging_dict

    
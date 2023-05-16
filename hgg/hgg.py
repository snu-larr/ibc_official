import copy
import numpy as np
from hgg.gcc_utils import gcc_load_lib, c_double
import torch

def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a - goal_b, ord=2)
def goal_concat(obs, goal):
	return np.concatenate([obs, goal], axis=0)

class TrajectoryPool:
	def __init__(self, pool_length, num_episodes=None):
		self.length = pool_length
		self.pool = []
		self.pool_init_state = []
		self.counter = 0
		if num_episodes in [None, 'none', 'None']:
			self.num_episodes = self.length
		else:
			self.num_episodes = num_episodes


	def insert(self, trajectory, init_state):
		if self.counter<self.length:
			self.pool.append(trajectory.copy())
			self.pool_init_state.append(init_state.copy())
		else:
			self.pool[self.counter%self.length] = trajectory.copy()
			self.pool_init_state[self.counter%self.length] = init_state.copy()
		self.counter += 1

	def pad(self):
		if self.counter>=self.length:
			return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
		pool = copy.deepcopy(self.pool)
		pool_init_state = copy.deepcopy(self.pool_init_state)
		# while len(pool)<self.length:
		while len(pool)<self.num_episodes:
			pool += copy.deepcopy(self.pool)
			pool_init_state += copy.deepcopy(self.pool_init_state)
		return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

class MatchSampler:    
	def __init__(self, goal_env, goal_eval_env, env_name, achieved_trajectory_pool, num_episodes,				
				add_noise_to_goal= False, cost_type='hgg_default', agent_type='forward', gamma=0.99, hgg_c=3.0, hgg_L=5.0, device = 'cuda',
				init_compute_type='proprioceptive', return_init_candidates_for_backward_proprioceptive = False,
				match_lib_path=None, sparse_reward_type=None, normalize_rl_obs = False,
				):
		# Assume goal env
		self.env = goal_env
		self.eval_env = goal_eval_env
		self.env_name = env_name
		self.vf = None
		self.critic = None
		self.policy = None
		self.add_noise_to_goal = add_noise_to_goal
		self.cost_type = cost_type
		self.agent_type= agent_type
		self.gamma = gamma
		self.hgg_c = hgg_c
		self.hgg_L = hgg_L
		self.device = device		
		self.init_compute_type = init_compute_type
		self.return_init_candidates_for_backward_proprioceptive = return_init_candidates_for_backward_proprioceptive
		self.sparse_reward_type = sparse_reward_type
		self.normalize_rl_obs = normalize_rl_obs
		
		self.total_cost = None
		self.total_forward_cost = None
		self.total_backward_cost = None

		
		self.success_threshold = {'sawyer_door' : 0.02,                                     
									'tabletop_manipulation' : 0.2,
									'fetch_push_ergodic' : 0.05,
									'fetch_pickandplace_ergodic' : 0.05,
									'fetch_reach_ergodic' : 0.05,									  
									'point_umaze' : 0.6,									
								}

		self.dim = np.prod(self.env.convert_obs_to_dict(self.env.reset())['achieved_goal'].shape)
		self.delta = self.success_threshold[env_name] #self.env.distance_threshold
		self.goal_distance = goal_distance

		self.length = num_episodes # args.episodes
		
		if self.agent_type=='forward':		
			init_goal = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['achieved_goal'].copy()
		elif self.agent_type=='backward':		
			init_goal = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['desired_goal'].copy()

		self.pool = np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
		
		
		if self.return_init_candidates_for_backward_proprioceptive:
			if self.agent_type=='forward':		
				if self.env_name in ['tabletop_manipulation', 'sawyer_door']: # pure_obs==ag
					init_goal_for_backward_proprioceptive = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['achieved_goal'].copy()
				elif self.env_name in ['fetch_pickandplace_ergodic', 'fetch_push_ergodic']:
					init_goal_for_backward_proprioceptive = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['observation'][..., :3].copy()
				elif self.env_name in ['fetch_reach_ergodic']:
					init_goal_for_backward_proprioceptive = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['observation'][..., :3].copy()
				elif self.env_name in ['point_umaze']:
					init_goal_for_backward_proprioceptive = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['observation'][..., :2].copy()
				else:
					raise NotImplementedError
			
			elif self.agent_type=='backward':		
				raise NotImplementedError

			self.pool_inits_for_backward_proprioceptive = np.tile(init_goal_for_backward_proprioceptive[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
		
		self.match_lib = gcc_load_lib(match_lib_path+'/hgg/cost_flow.c')
		
		self.achieved_trajectory_pool = achieved_trajectory_pool

		# estimating diameter
		self.max_dis = 0
		for i in range(1000):
			obs = self.env.convert_obs_to_dict(self.env.reset())
			dis = self.goal_distance(obs['achieved_goal'],obs['desired_goal'])
			if dis>self.max_dis: self.max_dis = dis
	
	def set_networks(self, vf=None, critic=None, policy=None):
		if vf is not None:
			assert critic is None and policy is None
			self.vf = vf
		elif critic is not None:
			assert policy is not None and vf is None
			self.critic = critic
			self.policy = policy
		

	def add_noise(self, pre_goal, noise_std=None, proprioceptive_only=False):
		goal = pre_goal.copy()		
		if noise_std is None: noise_std = self.delta
		
		if self.env_name=='sawyer_door':
			goal[4:6] += np.random.normal(0, noise_std, size=2)
			if proprioceptive_only:
				pass
			else:
				goal[6] = np.clip(goal[6], 0.1, 0.11) # door z position

		elif self.env_name =='tabletop_manipulation':
			goal[2:4] += np.random.normal(0, noise_std, size=2)
			goal[:4] = np.clip(goal[:4], -2.8, 2.8)
		elif self.env_name in ['fetch_pickandplace_ergodic', 'fetch_push_ergodic', 'fetch_reach_ergodic']:
			assert goal.shape[-1]==3
			goal += np.random.normal(0, noise_std, size=3)	
			if self.env_name=='fetch_push_ergodic':
				if proprioceptive_only:
					goal[2] = np.clip(goal[2], 0.42, 1.0) # prevent under the table
				else:
					goal[0] = np.clip(goal[0], 1.19786948, 1.49786948)
					goal[1] = np.clip(goal[1], 0.59894948, 0.89894948)
					goal[2] = np.clip(goal[2], 0.42, 0.43)
			elif self.env_name=='fetch_pickandplace_ergodic':
				if proprioceptive_only:
					goal[2] = np.clip(goal[2], 0.42, 1) # prevent under the table
				else:
					goal[0] = np.clip(goal[0], 1.19786948, 1.49786948)
					goal[1] = np.clip(goal[1], 0.59894948, 0.89894948)			
					goal[2] = np.clip(goal[2], 0.42, 1)
			elif self.env_name=='fetch_reach_ergodic':
				goal[2] = np.clip(goal[2], 0.42, 1) # prevent under the table
			else:
				raise NotImplementedError
					
			
			
		elif self.env_name =='point_umaze':
			assert goal.shape[-1]==2
			goal += np.random.normal(0, noise_std, size=2)
		
		else:
			raise NotImplementedError

		return goal.copy()

	def sample(self, idx, backward_proprioceptive=False):		
		if self.add_noise_to_goal:
			if self.env_name=='sawyer_door':
				noise_std = 0.005
			elif self.env_name in ['fetch_pickandplace_ergodic', 'fetch_push_ergodic', 'fetch_reach_ergodic']:
				noise_std = 0.03
			elif self.env_name=='tabletop_manipulation':
				noise_std = 0.1
			elif self.env_name=='point_umaze':
				noise_std = 0.3
			else:
				raise NotImplementedError('Should consider noise scale env by env')
			
			if backward_proprioceptive:
				return self.add_noise(self.pool_inits_for_backward_proprioceptive[idx], noise_std = noise_std, proprioceptive_only=True).astype(np.float32)
			else:
				return self.add_noise(self.pool[idx], noise_std = noise_std).astype(np.float32)
		else:
			if backward_proprioceptive:
				return self.pool_inits_for_backward_proprioceptive[idx].copy().astype(np.float32)
			else:
				return self.pool[idx].copy().astype(np.float32)

	# def find(self, goal):
	# 	res = np.sqrt(np.sum(np.square(self.pool-goal),axis=1))
	# 	idx = np.argmin(res)
	# 	if test_pool:
	# 		self.args.logger.add_record('Distance/sampler', res[idx])
	# 	return self.pool[idx].copy()

	def update(self, initial_goals, desired_goals):
		# list of ag(for earl env) or pure_obs(for non earl env), dg from env.reset() (used for target in hgg) 
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy(desired_goals)
			print('hgg use desired_goal from T* due to pool.counter==0')
			if self.return_init_candidates_for_backward_proprioceptive:
				self.pool_inits_for_backward_proprioceptive = copy.deepcopy(initial_goals) # [bs, dim]				
				# candidate_inits_for_backward_proprioceptive will be used as a goal for backward proprioceptive_only agent
				if self.env_name in ['tabletop_manipulation', 'sawyer_door']: # pure_obs==ag
					pass # object goal would be garbage
				elif self.env_name in ['fetch_pickandplace_ergodic', 'fetch_push_ergodic']: # pure_obs!=ag
					self.pool_inits_for_backward_proprioceptive = [data[..., :3] for data in self.pool_inits_for_backward_proprioceptive]
				elif self.env_name in ['fetch_reach_ergodic']: # pure_obs!=ag
					self.pool_inits_for_backward_proprioceptive = [data[..., :3] for data in self.pool_inits_for_backward_proprioceptive]
				elif self.env_name in ['point_umaze']: # pure_obs!=ag
					self.pool_inits_for_backward_proprioceptive = [data[..., :2] for data in self.pool_inits_for_backward_proprioceptive]
				else:
					raise NotImplementedError

			return
		 
		# pure_obs(in earl, same as ag. it is used for dg in hgg), init_obs(including_ag when non earl env)
		achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
		if self.return_init_candidates_for_backward_proprioceptive:
			candidate_inits_for_backward_proprioceptive = []
		candidate_goals = []
		candidate_edges = []
		candidate_id = []
		#########
		candidate_costs =[]
		candidate_forward_costs = []
		candidate_backward_costs = []

		# agent = self.args.agent
		achieved_value = []
		for i in range(len(achieved_pool)):
			 # maybe for all timesteps in an episode
			if self.env_name in ['tabletop_manipulation', 'sawyer_door']:
				if 'sawyer' in self.env_name:
					assert achieved_pool[i].shape[-1]==7
				else:
					assert achieved_pool[i].shape[-1]==6
				# achived_pool is achieved_goal
				obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for j in range(achieved_pool[i].shape[0])] # list of [dim] (len = ts)
			elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
				assert achieved_pool[i].shape[-1]==25
				# achived_pool is pure_obs -> should select related elements (object)
				obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j][3:6]) for j in range(achieved_pool[i].shape[0])] # list of [dim] (len = ts)
			elif self.env_name in ['fetch_reach_ergodic']:
				assert achieved_pool[i].shape[-1]==10
				# achived_pool is pure_obs -> should select related elements (gripper)
				obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j][:3]) for j in range(achieved_pool[i].shape[0])] # list of [dim] (len = ts)
			elif self.env_name in ['point_umaze']:
				assert achieved_pool[i].shape[-1]==7
				# achived_pool is pure_obs -> should select related elements (gripper)
				obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j][:2]) for j in range(achieved_pool[i].shape[0])] # list of [dim] (len = ts)
			
			else:
				raise NotImplementedError
			
			with torch.no_grad():				
				obs_t = torch.from_numpy(np.stack(obs, axis =0)).float().to(self.device) #[ts, dim]				
				if self.normalize_rl_obs:
					from ibc import normalize_obs
					obs_t = normalize_obs(obs_t, self.env_name, device=self.device)					

				if self.vf is not None:
					value = self.vf(obs_t).detach().cpu().numpy()[:,0]
				elif self.critic is not None and self.policy is not None:
					n_sample = 10
					tiled_obs_t = torch.tile(obs_t, (n_sample, 1, 1)).view((-1, obs_t.shape[-1])) #[ts, dim] -> [n_sample*ts, dim]
					dist = self.policy(obs_t) # obs : [ts, dim]
					action = dist.rsample((n_sample,)) # [n_sample, ts, dim]
					action = action.view((-1, action.shape[-1])) # [n_sample*ts, dim]					
					actor_Q1, actor_Q2 = self.critic(tiled_obs_t, action)
					actor_Q = torch.min(actor_Q1, actor_Q2).view(n_sample, -1, actor_Q1.shape[-1]) # [n_sample*ts, dim(1)] -> [n_sample, ts, dim(1)] 
					value = torch.mean(actor_Q, dim = 0).detach().cpu().numpy()[:,0] #[ts, dim(1)] -> [ts,]
			value = np.clip(value, -1.0/(1.0-self.gamma), 0)
			assert self.sparse_reward_type=='negative'
			achieved_value.append(value.copy())

		n = 0
		graph_id = {'achieved':[],'desired':[]}
		for i in range(len(achieved_pool)):
			n += 1
			graph_id['achieved'].append(n)
		for i in range(len(desired_goals)):
			n += 1
			graph_id['desired'].append(n)
		n += 1
		self.match_lib.clear(n)

		for i in range(len(achieved_pool)):
			self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
		for i in range(len(achieved_pool)):
			for j in range(len(desired_goals)): # args.episodes(=50)
				if self.cost_type=='hgg_default':
					if self.env_name in ['sawyer_door']: 
						assert achieved_pool[i].shape[-1] in [7, 10, 13], "ee(3), grip_state(1), obj(3), [ee_vel(3), obj_vel(3)]"
						assert initial_goals[j].shape[-1]==7 
						assert desired_goals[j].shape[-1]==7
						# only consider obj related states when computing the cost of res						
						res = np.sqrt(np.sum(np.square(achieved_pool[i][..., 4:7]-desired_goals[j][..., 4:7]),axis=1)) - achieved_value[i]/(self.hgg_L/self.max_dis/(1-self.gamma))
						
						forward_cost = np.min(np.sqrt(np.sum(np.square(achieved_pool[i][..., 4:7]-desired_goals[j][..., 4:7]),axis=1)))
						
						if self.init_compute_type=='object':							
							match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0][4:7], initial_goals[j][4:7])*self.hgg_c	# phi - phi_hat				
						elif self.init_compute_type=='proprioceptive':							
							match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0][:3], initial_goals[j][:3])*self.hgg_c	# phi - phi_hat				
							backward_cost = self.goal_distance(achieved_pool[i][0][:3], initial_goals[j][:3])
						elif self.init_compute_type=='object_proprioceptive':
							match_dis = np.min(res)+self.goal_distance(np.concatenate([achieved_pool[i][0][:3], achieved_pool[i][0][4:7]], axis =-1), np.concatenate([initial_goals[j][:3], initial_goals[j][4:7]], axis =-1))*self.hgg_c	# phi - phi_hat				
						else:
							raise NotImplementedError

					elif self.env_name == 'tabletop_manipulation':
						assert achieved_pool[i].shape[-1]==6
						assert initial_goals[j].shape[-1]==6 
						assert desired_goals[j].shape[-1]==6
						# consider obj & gripper related states when computing the cost of res						
						res = np.sqrt(np.sum(np.square(achieved_pool[i][..., :4]-desired_goals[j][..., :4]),axis=1)) - achieved_value[i]/(self.hgg_L/self.max_dis/(1-self.gamma))
						
						forward_cost = np.min(np.sqrt(np.sum(np.square(achieved_pool[i][..., :4]-desired_goals[j][..., :4]),axis=1)))
						
						if self.init_compute_type=='object':							
							match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0][2:4], initial_goals[j][2:4])*self.hgg_c	# phi - phi_hat				
						elif self.init_compute_type=='proprioceptive':							
							match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0][:2], initial_goals[j][:2])*self.hgg_c	# phi - phi_hat				
							backward_cost = self.goal_distance(achieved_pool[i][0][:2], initial_goals[j][:2])
						elif self.init_compute_type=='object_proprioceptive':
							match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0][:4], initial_goals[j][:4])*self.hgg_c	# phi - phi_hat				
						else:
							raise NotImplementedError
					
					elif self.env_name in ['fetch_push_ergodic', 'fetch_pickandplace_ergodic']:
						assert achieved_pool[i].shape[-1]==25
						assert initial_goals[j].shape[-1]==25 
						assert desired_goals[j].shape[-1]==3
						# only consider obj related states when computing the cost of res						
						res = np.sqrt(np.sum(np.square(achieved_pool[i][..., 3:6]-desired_goals[j][..., :3]),axis=1)) - achieved_value[i]/(self.hgg_L/self.max_dis/(1-self.gamma))
						
						forward_cost = np.min(np.sqrt(np.sum(np.square(achieved_pool[i][..., 3:6]-desired_goals[j][..., :3]),axis=1)))

						if self.init_compute_type=='object':							
							match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0][3:6], initial_goals[j][3:6])*self.hgg_c	# phi - phi_hat				
						elif self.init_compute_type=='proprioceptive':							
							match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0][:3], initial_goals[j][:3])*self.hgg_c	# phi - phi_hat				
							backward_cost = self.goal_distance(achieved_pool[i][0][:3], initial_goals[j][:3])
						elif self.init_compute_type=='object_proprioceptive':
							match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0][:6], initial_goals[j][:6])*self.hgg_c	# phi - phi_hat				
						else:
							raise NotImplementedError

					elif self.env_name in ['fetch_reach_ergodic']:						
						assert achieved_pool[i].shape[-1]==10
						assert initial_goals[j].shape[-1]==10
						assert desired_goals[j].shape[-1]==3

						# only consider gripper related states when computing the cost of res
						res = np.sqrt(np.sum(np.square(achieved_pool[i][..., :3]-desired_goals[j][..., :3]),axis=1)) - achieved_value[i]/(self.hgg_L/self.max_dis/(1-self.gamma))
						
						forward_cost = np.min(np.sqrt(np.sum(np.square(achieved_pool[i][..., :3]-desired_goals[j][..., :3]),axis=1)))
						
						if self.init_compute_type=='object':							
							raise NotImplementedError
						elif self.init_compute_type=='proprioceptive':							
							match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0][:3], initial_goals[j][:3])*self.hgg_c	# phi - phi_hat				
							backward_cost = self.goal_distance(achieved_pool[i][0][:3], initial_goals[j][:3])
						elif self.init_compute_type=='object_proprioceptive':
							raise NotImplementedError
						else:
							raise NotImplementedError

					elif self.env_name in ['point_umaze']:
						assert achieved_pool[i].shape[-1]==7
						assert initial_goals[j].shape[-1]==7
						assert desired_goals[j].shape[-1]==2 
						# only consider agent related states when computing the cost of res
						res = np.sqrt(np.sum(np.square(achieved_pool[i][..., :2]-desired_goals[j][..., :2]),axis=1)) - achieved_value[i]/(self.hgg_L/self.max_dis/(1-self.gamma))
						
						forward_cost = np.min(np.sqrt(np.sum(np.square(achieved_pool[i][..., :2]-desired_goals[j][..., :2]),axis=1)))

						if self.init_compute_type=='object':							
							raise NotImplementedError
						elif self.init_compute_type=='proprioceptive':							
							match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0][:2], initial_goals[j][:2])*self.hgg_c	# phi - phi_hat				
							backward_cost = self.goal_distance(achieved_pool[i][0][:2], initial_goals[j][:2])
						elif self.init_compute_type=='object_proprioceptive':
							raise NotImplementedError
						else:
							raise NotImplementedError
					
					else:
						res = np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1)) - achieved_value[i]/(self.hgg_L/self.max_dis/(1-self.gamma))
						forward_cost = np.min(np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1)))
						match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0], initial_goals[j])*self.hgg_c
						backward_cost = self.goal_distance(achieved_pool[i][0], initial_goals[j])
				
				else:
					raise NotImplementedError


				match_idx = np.argmin(res)

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				if self.env_name in ['tabletop_manipulation', 'sawyer_door']: # pure_obs==ag
					candidate_goals.append(achieved_pool[i][match_idx])
				elif self.env_name in ['fetch_pickandplace_ergodic', 'fetch_push_ergodic']: # pure_obs!=ag
					candidate_goals.append(achieved_pool[i][match_idx][3:6]) # consider object goal
				elif self.env_name in ['fetch_reach_ergodic']: # pure_obs!=ag
					candidate_goals.append(achieved_pool[i][match_idx][:3])
				elif self.env_name in ['point_umaze']: # pure_obs!=ag
					candidate_goals.append(achieved_pool[i][match_idx][:2])
				else:
					raise NotImplementedError

				# candidate_goals.append(achieved_pool[i][match_idx]) # original


				if self.return_init_candidates_for_backward_proprioceptive:
					# candidate_inits_for_backward_proprioceptive will be used as a goal for backward proprioceptive_only agent
					if self.env_name in ['tabletop_manipulation', 'sawyer_door']: # pure_obs==ag
						candidate_inits_for_backward_proprioceptive.append(achieved_pool[i][0]) # object goal would be garbage
					elif self.env_name in ['fetch_pickandplace_ergodic', 'fetch_push_ergodic']: # pure_obs!=ag
						candidate_inits_for_backward_proprioceptive.append(achieved_pool[i][0][:3]) # consider gripper goal
					elif self.env_name in ['fetch_reach_ergodic', 'fetch_reach_ergodic']: # pure_obs!=ag
						candidate_inits_for_backward_proprioceptive.append(achieved_pool[i][0][:3])
					elif self.env_name in ['point_umaze']: # pure_obs!=ag
						candidate_inits_for_backward_proprioceptive.append(achieved_pool[i][0][:2])
					else:
						raise NotImplementedError
					

				candidate_edges.append(edge)
				candidate_id.append(j)
				########
				candidate_costs.append(match_dis)
				candidate_forward_costs.append(forward_cost)
				candidate_backward_costs.append(backward_cost)

		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0,n)
		
		assert match_count==self.length, 'match count {} self.length :{}'.format(match_count, self.length)

		explore_goals = [0]*self.length
		#######
		explore_costs = [0]*self.length
		explore_forward_costs = [0]*self.length
		explore_backward_costs = [0]*self.length

		if self.return_init_candidates_for_backward_proprioceptive:
			explore_inits_for_backward_proprioceptive = [0]*self.length
		
		

		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i])==1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
				########
				explore_costs[candidate_id[i]] = candidate_costs[i].copy()
				explore_forward_costs[candidate_id[i]] = candidate_forward_costs[i].copy()
				explore_backward_costs[candidate_id[i]] = candidate_backward_costs[i].copy()

				if self.return_init_candidates_for_backward_proprioceptive:
					explore_inits_for_backward_proprioceptive[candidate_id[i]] = candidate_inits_for_backward_proprioceptive[i].copy()
		
		self.total_cost = np.stack(explore_costs)
		self.total_forward_cost = np.stack(explore_forward_costs)
		self.total_backward_cost = np.stack(explore_backward_costs)

		assert len(explore_goals)==self.length
		self.pool = np.array(explore_goals)
		if self.return_init_candidates_for_backward_proprioceptive:
			self.pool_inits_for_backward_proprioceptive = np.array(explore_inits_for_backward_proprioceptive)

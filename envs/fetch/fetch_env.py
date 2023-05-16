import numpy as np

from gym_robotics.envs import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments."""

    def __init__(
        self,
        model_path,
        n_substeps,
        gripper_extra_height,
        block_gripper,
        has_object,
        target_in_the_air,
        target_offset,
        obj_range,
        target_range,
        distance_threshold,
        initial_qpos,
        reward_type,
        full_state_goal=False, 
        fix_initial_object = False,
        
        
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        
        
        self.full_state_goal = full_state_goal
        self.proprioceptive_only = False
        self.fix_initial_object = fix_initial_object
        

        if has_object:
            if block_gripper and (not target_in_the_air):
                self.env_type = 'push'
            elif (not block_gripper) and target_in_the_air:
                self.env_type = 'pickandplace'
            else:
                raise NotImplementedError

        super().__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=4,
            initial_qpos=initial_qpos,
        )
    
    # ----------------------------
    def set_proprioceptive_only(self, proprioceptive_only):
        self.proprioceptive_only = proprioceptive_only
    
    def set_tstar_states(self, hand_pos=None, obj_pos=None):
    # self.sim.model.site_pos[self.model.site_name2id('goal')] = self._handle_goal
      if hand_pos is not None:
        for i in range(hand_pos.shape[0]):
          self.sim.model.site_pos[self.sim.model.site_name2id('hand_'+str(i+1))] = hand_pos[i]
      if obj_pos is not None:
        for i in range(obj_pos.shape[0]):
          self.sim.model.site_pos[self.sim.model.site_name2id('obj_'+str(i+1))] = obj_pos[i]

    # EARL env methods
    # ----------------------------

    def reset_goal(self, goal=None, add_noise = False):
        if goal is None:
            goal = self._sample_goal()
        
        if add_noise:
            goal = goal + np.random.normal(scale=0.002, size = goal.shape)
            
        self.goal = goal

        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        if self.full_state_goal:
            if self.proprioceptive_only:
                self.sim.model.site_pos[self.sim.model.site_name2id("target0")] = self.goal[:3] - sites_offset[0]
            else:
                self.sim.model.site_pos[self.sim.model.site_name2id("target0")] = self.goal[-3:] - sites_offset[0]
        else:
            self.sim.model.site_pos[self.sim.model.site_name2id("target0")] = self.goal - sites_offset[0]

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        if self.full_state_goal:
            if self.proprioceptive_only: # consider only gripper reaching
                d = goal_distance(achieved_goal[..., :3], goal[..., :3])
            else:
                d = goal_distance(achieved_goal[..., -3:], goal[..., -3:])
        else:
            d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
            self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos("object0")
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
            # velocities
            object_velp = self.sim.data.get_site_xvelp("object0") * dt
            object_velr = self.sim.data.get_site_xvelr("object0") * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:            
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )
        if self.full_state_goal:
            assert self.has_object
            return {
                "observation": obs.copy(),
                "achieved_goal": np.concatenate([grip_pos.copy(), object_pos.copy()], axis =-1),
                "desired_goal": self.goal.copy(),
            }
        else:
            if self.proprioceptive_only: # consider only gripper reaching
                achieved_goal = grip_pos.copy()
                
            return {
                "observation": obs.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.copy(),
            }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.5 # 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        if self.full_state_goal:
            if self.proprioceptive_only:
                self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
            else:
                self.sim.model.site_pos[site_id] = self.goal[-3:] - sites_offset[0]
        else:
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpx = self.sim.data.get_joint_qpos("object0:joint_px")
            object_qpy = self.sim.data.get_joint_qpos("object0:joint_py")
            object_qpz = self.sim.data.get_joint_qpos("object0:joint_pz")
            object_qrxyz = self.sim.data.get_joint_qpos("object0:joint_rxyz")
            object_qpos = np.concatenate([np.atleast_1d(object_qpx), np.atleast_1d(object_qpy), np.atleast_1d(object_qpz), object_qrxyz])
            assert object_qpos.shape == (7,)
            if self.fix_initial_object:
                pass
            else:
                object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos("object0:joint_px", object_qpos[0])
            self.sim.data.set_joint_qpos("object0:joint_py", object_qpos[1])

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
            
            if self.full_state_goal:
                if self.proprioceptive_only:
                    if self.env_type == 'push':
                        gripper_goal = goal.copy()
                        if self.np_random.uniform() < 0.5:
                            gripper_goal[2] += self.np_random.uniform(0, 0.45)                        
                        goal = np.concatenate([gripper_goal, goal], axis =-1) #[grip, obj]
                    elif self.env_type == 'pickandplace':                        
                        goal = np.concatenate([goal, goal], axis =-1)
                    else:
                        raise NotImplementedError

                else: # grip goal == obj_goal
                    goal = np.concatenate([goal, goal], axis =-1)
            else:
                if self.proprioceptive_only: # consider only gripper reaching
                    if (not self.target_in_the_air) and self.np_random.uniform() < 0.5:
                        goal[2] += self.np_random.uniform(0, 0.45)
                else:
                    pass
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        if self.full_state_goal:
            if self.proprioceptive_only: # consider only gripper reaching
                d = goal_distance(achieved_goal[..., :3], desired_goal[..., :3])
            else:
                d = goal_distance(achieved_goal[..., -3:], desired_goal[..., -3:])
        else:
            d = goal_distance(achieved_goal, desired_goal)
        
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self.sim.data.get_site_xpos("robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos("object0")[2]

    def render(self, mode="human", width=500, height=500):
        return super().render(mode, width, height)

import numpy as np
import os
from gym import utils
from .fetch_env import FetchEnv
import gym_robotics

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/fetch/ergodic_pick_and_place2.xml")


class FetchPickAndPlaceErgodicEnv2(FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse", full_state_goal=False, reset_at_goal=False):
        '''
        compared to v1:
        i) virtual limit of object0 position is reduced by 0.04 (along x, y axis)
        ii) x,y joint limit object0 is hardened (solreflimit, solimplimit)
        iii) initial position of object0 is "almost" fixed
        '''
        
        self.reset_at_goal = reset_at_goal

        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint_px": 1.25,
            "object0:joint_py": 0.53,
            "object0:joint_pz": 0.4,
            "object0:joint_rxyz": [1.0, 0.0, 0.0, 0.0],
        }
        FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            full_state_goal=full_state_goal,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        if self.reset_at_goal:
            if self.has_object: 
                # get initial config of object
                object_qpx = self.sim.data.get_joint_qpos("object0:joint_px")
                object_qpy = self.sim.data.get_joint_qpos("object0:joint_py")
                object_qpz = self.sim.data.get_joint_qpos("object0:joint_pz")
                object_qrxyz = self.sim.data.get_joint_qpos("object0:joint_rxyz")

                init_obj_xpos = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                init_obj_xpos += self.target_offset
                init_obj_xpos[2] = self.height_offset
                
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    init_obj_xpos[2] += self.np_random.uniform(0, 0.45)
                    
                
                gripper_target = init_obj_xpos.copy() + np.array([0, 0, 0.015])
                gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
                self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
                self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
                
                action = np.array([0, 0, 0, 1.0])
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
                    self._step_callback()
                    

                action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

                # Apply action to simulation.
                # gym_robotics.envs.utils.ctrl_set_action(self.sim, action)
                # gym_robotics.envs.utils.mocap_set_action(self.sim, action)

                
                for _ in range(5): # move the end effector with opened gripper
                    self.sim.step()

                object_qpos = np.concatenate([np.atleast_1d(object_qpx), np.atleast_1d(object_qpy), np.atleast_1d(object_qpz), object_qrxyz])
                assert object_qpos.shape == (7,)
                custom_offset = np.array([-0.025, -0.025, -0.025])                    
                for _ in range(1): # set object position with closed gripper                    
                    object_qpos[:3] = init_obj_xpos + custom_offset
                    self.sim.data.set_joint_qpos("object0:joint_px", object_qpos[0])
                    self.sim.data.set_joint_qpos("object0:joint_py", object_qpos[1])
                    self.sim.data.set_joint_qpos("object0:joint_pz", object_qpos[2])
                    
                    gripper_ctrl = np.array([-1.0, -1.0])
                    action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
                    # Apply action to simulation.
                    gym_robotics.envs.utils.ctrl_set_action(self.sim, action)
                    gym_robotics.envs.utils.mocap_set_action(self.sim, action)
                    self.sim.step()

        else:
            # "Fix" start position of object.
            if self.has_object:
                object_xpos = np.array([1.43, 0.76]) + self.np_random.uniform(-0.025, 0.025, size=2)
                object_qpx = self.sim.data.get_joint_qpos("object0:joint_px")
                object_qpy = self.sim.data.get_joint_qpos("object0:joint_py")
                object_qpz = self.sim.data.get_joint_qpos("object0:joint_pz")
                object_qrxyz = self.sim.data.get_joint_qpos("object0:joint_rxyz")
                object_qpos = np.concatenate([np.atleast_1d(object_qpx), np.atleast_1d(object_qpy), np.atleast_1d(object_qpz), object_qrxyz])
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos("object0:joint_px", object_qpos[0])
                self.sim.data.set_joint_qpos("object0:joint_py", object_qpos[1])

        self.sim.forward()
        return True
    
    def _sample_goal(self):
        if self.reset_at_goal:
            if self.has_object:
                object_xpos = np.array([1.43, 0.76]) + self.np_random.uniform(-0.025, 0.025, size=2)
                goal = np.concatenate([object_xpos, np.array([self.height_offset])], axis =-1)            
            
            return goal.copy()

        else:
            return super()._sample_goal()
        
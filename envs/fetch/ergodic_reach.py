import os
from gym import utils
from .fetch_env import FetchEnv
import numpy as np

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/fetch/reach.xml")


class FetchReachErgodicEnv(FetchEnv, utils.EzPickle): # is identical to FetchReachEnv
    def __init__(self, reward_type="sparse", reset_at_goal=False):
        
        self.reset_at_goal = reset_at_goal

        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)
    
    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        if self.reset_at_goal:
            if self.has_object: 
                raise NotImplementedError
            else:
                init_gripper_xpos = self._sample_goal()                
                
                gripper_target = init_gripper_xpos.copy()
                gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
                self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
                self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
                
                if self.block_gripper:
                    self._step_callback()
                    
                for _ in range(5): # move the end effector with opened gripper
                    self.sim.step()

            self.sim.forward()
            return True
        else:
            return super()._reset_sim()
            

    
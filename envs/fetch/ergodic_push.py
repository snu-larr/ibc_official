import os
from gym import utils
from .fetch_env import FetchEnv


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/fetch/ergodic_push.xml")


class FetchPushErgodicEnv(FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse", full_state_goal=False):
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
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            full_state_goal=full_state_goal,      
            
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

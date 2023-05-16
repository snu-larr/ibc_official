import gym, gym_robotics
import os
gym_base = os.path.dirname(gym_robotics.__file__)
base = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(base, 'assets/stls')):
    try:
        os.remove(os.path.join(base, 'assets/stls'))
    except FileNotFoundError:
        pass
    os.symlink(os.path.join(gym_base, 'envs/assets/stls'), os.path.join(base, 'assets/stls'))
if not os.path.isdir(os.path.join(base, 'assets/textures')):
    try:
        os.remove(os.path.join(base, 'assets/textures'))
    except FileNotFoundError:
        pass
    os.symlink(os.path.join(gym_base, 'envs/assets/textures'), os.path.join(base, 'assets/textures'))

from .fetch.ergodic_pick_and_place import FetchPickAndPlaceErgodicEnv
from .fetch.ergodic_push import FetchPushErgodicEnv
from .fetch.ergodic_reach import FetchReachErgodicEnv

from .fetch.ergodic_push2 import FetchPushErgodicEnv2
from .fetch.ergodic_pick_and_place2 import FetchPickAndPlaceErgodicEnv2

from .mujoco_maze.point_umaze_goalenv import PointUMazeGoalEnv
from .mujoco_maze.ant_umaze_goalenv import AntUMazeGoalEnv
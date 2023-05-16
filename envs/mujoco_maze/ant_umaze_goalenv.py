import numpy as np
from typing import List

from .maze_env import MazeEnv, MazeGoalEnv
from .maze_task import MazeTask, MazeGoal, MazeCell
from .ant import AntEnv


class GoalEnvUMaze(MazeTask):
    @property
    def goal(self):
        assert len(self.goals) == 1
        return self.goals[0].pos

    @property
    def distance_threshold(self):
        assert len(self.goals) == 1
        return self.goals[0].threshold
    
    def __init__(self, scale: float, reward_type: str) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.0 * self.scale]))]
        self.reward_type = reward_type

    def _sample_goal(self) -> np.ndarray:
        # TODO: sample goals uniformly in free space?
        goal = np.array([0.0, 2.0 * self.scale])
        return goal

    def sample_goals(self, goal=None) -> bool:
        if goal is None:
            goal = self._sample_goal()
        self.goals = [MazeGoal(self._sample_goal())]
        return True

    def termination(self, obs: np.ndarray) -> bool:
        return False

    def reward(self, obs: np.ndarray) -> float:
        return self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ]

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict):
        # Compute distance between goal and the achieved goal.
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def get_achieved_goal(self, observation: np.ndarray) -> np.ndarray: # goal space transformation (o -> g)
        return observation[: self.goals[0].dim]


class AntUMazeGoalEnv(MazeGoalEnv): # goal env implementation
    def __init__(self, reward_type="sparse"):
        MazeEnv.__init__(
            self,
            model_cls=AntEnv,
            maze_task=GoalEnvUMaze,
            maze_size_scaling=GoalEnvUMaze.MAZE_SIZE_SCALING.point,
            inner_reward_scaling=GoalEnvUMaze.INNER_REWARD_SCALING,
            task_kwargs=dict(reward_type=reward_type),
        )
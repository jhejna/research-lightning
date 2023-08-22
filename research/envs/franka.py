"""
This environment is for use with a real panda robot and polymetis.
"""

import time
from abc import abstractmethod, abstractproperty
from typing import Optional

import gym
import numpy as np
import torch
from polymetis import GripperInterface, RobotInterface
from scipy.spatial.transform import Rotation

__all__ = ["FrankaEnv", "FrankaReach"]


class Controller(object):
    # Joint limits from:
    # https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/conf/robot_client/franka_hardware.yaml

    EE_LOW = np.array([0.1, -0.4, -0.05, -np.pi, -np.pi, -np.pi], dtype=np.float32)
    EE_HIGH = np.array([1.0, 0.4, 1.0, 1.0, np.pi, np.pi, np.pi], dtype=np.float32)
    JOINT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
    JOINT_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)
    HOME = np.array([0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0], dtype=np.float32)

    def __init__(self, ip_address: str = "localhost", control_hz=10.0):
        self.robot = RobotInterface(ip_address=ip_address)
        self.robot.set_home_pose(torch.Tensor(self.HOME))
        self.gripper = GripperInterface(ip_address=ip_address)
        if hasattr(self.gripper, "metadata") and hasattr(self.gripper.metadata, "max_width"):
            # Should grab this from robotiq2f
            self._max_gripper_width = self.gripper.metadata.max_width
        else:
            self._max_gripper_width = 0.08  # FrankaHand Value
        self._updated = True
        self._running = False
        self._control_hz = control_hz

    @property
    def action_space(self):
        # Returns the action space with the gripper
        low = np.concatenate((self.robot_action_space.low, [0]))
        high = np.concatenate((self.robot_action_space.high, [1]))
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "joint_positions": gym.spaces.Box(low=self.JOINT_LOW, high=self.JOINT_HIGH, dtype=np.float32),
                "joint_velocities": gym.spaces.Box(
                    low=-np.inf * self.JOINT_LOW, high=np.inf * self.JOINT_HIGH, dtype=np.float32
                ),
                "ee_pos": gym.spaces.Box(low=self.EE_LOW[:3], high=self.EE_HIGH[:3], dtype=np.float32),
                "ee_quat": gym.spaces.Box(low=np.zeros(4), high=np.ones(4), dtype=np.float32),
                "gripper_pos": gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32),
            }
        )

    def get_state(self):
        """
        This function simultaneously:
        1. Marks the current state (ONLY CALL WHEN RETURNING FINAL OBS)
        2. Returns it as an observation dict.
        When used improperly, the updated function will be called
        """
        assert self._updated
        robot_state = self.robot.get_robot_state()
        ee_pos, ee_quat = self.robot.robot_model.forward_kinematics(torch.Tensor(robot_state.joint_positions))
        gripper_state = self.gripper.get_state()
        gripper_pos = 1 - (gripper_state.width / self._max_gripper_width)
        self._updated = False
        self.state = dict(
            joint_positions=np.array(robot_state.joint_positions, dtype=np.float32),
            joint_velocities=np.array(robot_state.joint_velocities, dtype=np.float32),
            ee_pos=ee_pos.numpy(),
            ee_quat=ee_quat.numpy(),
            gripper_pos=gripper_pos,
        )
        return self.state

    def update(self, action):
        assert not self._updated
        # Step the controller
        robot_action, gripper_action = action[:-1], action[-1]
        # Make sure neither commands block.
        self.update_robot(robot_action)
        self.update_gripper(gripper_action, blocking=True)
        self._updated = True

    def reset(self, randomize: bool = False):
        if self._running:
            self.robot.terminate_current_policy()
        self.update_gripper(0, blocking=False)  # Close the gripper
        self.robot.go_home()
        if randomize:
            # Get the current position and then add some noise to it
            joint_positions = self.robot.get_joint_positions()
            # Update the desired joint positions
            high = 0.1 * np.ones(self.JOINT_LOW.shape[0], dtype=np.float32)
            noise = np.random.uniform(low=-high, high=high)
            randomized_joint_positions = np.array(joint_positions, dtype=np.float32) + noise
            self.robot.move_to_joint_positions(torch.from_numpy(randomized_joint_positions))
        self.start_robot()
        self._running = True
        self._updated = True

    def update_gripper(self, gripper_action, blocking=False):
        # We always run the gripper in absolute position
        gripper_action = max(min(gripper_action, 1), 0)
        self.gripper.goto(
            width=self._max_gripper_width * (1 - gripper_action), speed=0.1, force=0.01, blocking=blocking
        )

    @abstractproperty
    def robot_action_space(self):
        raise NotImplementedError

    @abstractmethod
    def start_robot(self):
        pass

    @abstractmethod
    def update_robot(self, robot_action):
        pass


class CartesianPositionController(Controller):
    @property
    def robot_action_space(self):
        return gym.spaces.Box(low=self.EE_LOW, high=self.EE_HIGH, dtype=np.float32)

    def start_robot(self):
        self.robot.start_cartesian_impedance()

    def update_robot(self, action):
        action = np.clip(action, self.EE_LOW, self.EE_HIGH)
        pos, ori = action[:3], Rotation.from_euler("xyz", action[3:]).as_quat()
        self.robot.update_desired_ee_pose(torch.from_numpy(pos), torch.from_numpy(ori))


class CartesianDeltaController(Controller):
    """
    Note that the action space here is smaller since we use deltas in Euler!
    this is consistent with other delta action spaces.
    """

    def __init__(self, *args, max_delta: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_delta is not None:
            self._max_delta = max_delta
        else:
            # Define dynamically based on control hz.
            # What do we want the max velocity to be.
            self._max_delta = 0.1 / self._control_hz

    @property
    def robot_action_space(self):
        high = self._max_delta * np.ones(6, dtype=np.float32)
        # We aim to construct the following action space for 2 Hz or _max_delta = 0.05
        # x: 0.05, y 0.05, z 0.05, rot1 0.2 rot2 0.2 rot3 0.2
        high[3:] = 4 * self._max_delta
        return gym.spaces.Box(low=-1 * high, high=high, dtype=np.float32)

    def start_robot(self):
        self.robot.start_cartesian_impedance()

    def update_robot(self, action):
        action = np.clip(action, self.robot_action_space.low, self.robot_action_space.high)
        delta_pos, delta_ori = action[:3], action[3:]
        new_pos = torch.from_numpy(self.state["ee_pos"] + delta_pos).float()
        # TODO: this can be made much faster using purpose build methods instead of scipy.
        old_rot = Rotation.from_quat(self.state["ee_quat"])
        delta_rot = Rotation.from_euler("xyz", delta_ori)
        new_rot = delta_rot * old_rot
        new_quat = torch.from_numpy(new_rot.as_quat()).float()
        self.robot.update_desired_ee_pose(new_pos, new_quat)


class JointPositionController(Controller):
    @property
    def robot_action_space(self):
        return gym.spaces.Box(low=self.JOINT_LOW, high=self.JOINT_HIGH, dtype=np.float32)

    def start_robot(self):
        self.robot.start_joint_impedance()

    def update_robot(self, action):
        action = np.clip(action, self.JOINT_LOW, self.JOINT_HIGH)
        self.robot.update_desired_joint_positions(torch.from_numpy(action))


class JointDeltaController(Controller):
    def __init__(self, *args, max_delta: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_delta = max_delta

    @property
    def robot_action_space(self):
        high = self._max_delta * np.ones(self.JOINT_LOW.shape[0], dtype=np.float32)
        return gym.spaces.Box(low=-1 * high, high=high, dtype=np.float32)

    def start_robot(self):
        self.robot.start_joint_impedance()

    def update_robot(self, action):
        action = np.clip(action, self.robot_action_space.low, self.robot_action_space.high)
        new_joint_positions = self.state["joint_positions"] + action
        self.robot.update_desired_joint_positions(torch.from_numpy(new_joint_positions))


def precise_wait(t_end: float, slack_time: float = 0.001):
    t_start = time.time()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time.time() < t_end:
            pass
    else:
        print("[Franka] Warning: latentcy larger than desired control hz.")
    return


class FrankaEnv(gym.Env):
    """
    A simple Gym Environment for the Franka robots controlled via PolyMetis.
    TODO: support for images etc.
    """

    def __init__(
        self,
        ip_address: str = "localhost",
        controller: str = "cartesian_delta",
        control_hz: float = 10.0,
        horizon: str = 500,
    ):
        self.controller = {
            "cartesian_position": CartesianPositionController,
            "cartesian_delta": CartesianDeltaController,
            "joint_position": JointPositionController,
            "joint_delta": JointDeltaController,
        }[controller](ip_address=ip_address)
        # self.action_space = self.controller.action_space
        # Add the action space limits.
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.controller.action_space.shape, dtype=np.float32)
        # TODO: update later to modify in addition to proprio space with cameras etc.
        self.observation_space = self.controller.observation_space

        self.horizon = horizon
        self._max_episode_steps = horizon
        self.control_hz = float(control_hz)
        self._time = None  # Set time to None.
        self._steps = 0

    def step(self, action):
        # Immediately update with the action
        low, high = self.controller.action_space.low, self.controller.action_space.high
        unscaled_action = low + (0.5 * (action + 1.0) * (high - low))
        self.controller.update(unscaled_action)

        # Wait until we have 15hz since last time.
        precise_wait(self._time + 1 / self.control_hz)
        self._time = time.time()
        self._steps += 1
        done = self._steps == self.horizon
        state = self.controller.get_state()
        return state, 0, done, dict(discount=1.0)

    def reset(self):
        self.controller.reset(randomize=True)
        state = self.controller.get_state()
        self._steps = 0
        # start the timer.
        self._time = time.time()
        return state


class FrankaReach(FrankaEnv):
    """
    A simple environment where the goal is for the Franka to reach a specific end effector position.
    """

    def __init__(self, *args, goal_position=(0.5, 0.25, 0.5), horizon=100, **kwargs):
        self._goal_position = np.array(goal_position)
        super().__init__(*args, horizon=horizon, **kwargs)

    def step(self, action):
        state, reward, done, info = super().step(action)
        goal_distance = np.linalg.norm(state["ee_pos"] - self._goal_position)
        reward = -0.1 * goal_distance
        print(state["ee_pos"], reward)
        info["goal_distance"] = goal_distance
        return state, reward, done, info

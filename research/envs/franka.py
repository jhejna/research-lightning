"""
This environment is for use with a real panda robot and polymetis.
"""

import time
from abc import abstractmethod, abstractproperty
from typing import Dict, Optional

import gym
import numpy as np
from polymetis import GripperInterface, RobotInterface

__all__ = ["FrankaEnv", "FrankaReach"]


class Controller(object):
    # Joint limits from:
    # https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/conf/robot_client/franka_hardware.yaml

    EE_LOW = np.array([0.1, -0.4, -0.05, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    EE_HIGH = np.array([1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    JOINT_LOW = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159], dtype=np.float32)
    JOINT_HIGH = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], dtype=np.float32)

    def __init__(self, ip_address: str = "localhost"):
        self.robot = RobotInterface(ip_address=ip_address)
        self.gripper = GripperInterface()
        self._max_gripper_width = self._gripper.metadata.max_width
        self._updated = True

    @property
    def action_space(self):
        # Returns the action space with the gripper
        low = np.concatenate((self.robot_action_space.low, [0]))
        high = np.concatenate((self.robot_action_space.high, [1]))
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def get_state(self):
        """
        This function simultaneously:
        1. Marks the current state (ONLY CALL WHEN RETURNING FINAL OBS)
        2. Returns it as an observation dict.
        When used improperly, the updated function will be called
        """
        assert self._updated
        robot_state = self.robot.get_robot_state()
        ee_pos, ee_quat = self.robot.robot_model.forward_kinematics(robot_state.joint_positions)
        gripper_state = self.gripper.get_state()
        gripper_pos = 1 - (gripper_state.width / self._max_gripper_width)
        self._updated = False
        self.state = dict(
            joint_positions=robot_state.joint_positions,
            joint_velocities=robot_state.joint_velocities,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            gripper_pos=gripper_pos,
        )
        return self.state

    def update(self, action):
        assert not self._updated
        # Step the controller
        robot_action, gripper_action = action[:-1], action[-1]
        # Make sure neither commands block.
        self.update_robot(robot_action)
        self.update_gripper(gripper_action, blocking=False)
        self._updated = True

    def reset(self, randomize: bool = False):
        self.robot.terminate_current_policy()
        self.update_gripper(0, blocking=True)  # Close the gripper
        self.robot.go_home()
        if randomize:
            # Get the current position and then add some noise to it
            joint_positions = self.robot.get_joint_positions()
            # Update the desired joint positions
            high = 0.05 * np.ones(self.JOINT_LOW)
            noise = np.random.uniform(low=-high, high=high, dtype=joint_positions.dtype)
            self.robot.move_to_joint_positions(joint_positions + noise)
        self.start_robot()

    def update_gripper(self, gripper_action, blocking=False):
        # We always run the gripper in absolute position
        gripper_action = max(min(gripper_action, 1), 0)
        self.gripper.goto(
            width=self._max_gripper_width * (1 - gripper_action), speed=0.05, force=0.1, blocking=blocking
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
        action = np.clip(action, low=self.EE_LOW, high=self.EE_HIGH)
        pos, ori = action[:3], action[3:]
        self.robot.update_desired_ee_pos(pos, ori)


class CartesianDeltaController(Controller):
    def __init__(self, *args, max_delta: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_delta = max_delta

    @property
    def robot_action_space(self):
        high = self._max_delta * np.ones(self.EE_LOW.shape[0], dtype=np.float32)
        return gym.spaces.Box(low=-1 * high, high=high, dtype=np.float32)

    def start_robot(self):
        self.robot.start_cartesian_impedance()

    def update_robot(self, action):
        action = np.clip(action, low=self.robot_action_space.low, high=self.robot_action_space.high)
        delta_pos, delta_ori = action[:3], action[3:]
        # TODO: I'm not sure if direct sum is the correct way to add two orientations.
        # I think this is wrong, but going with it for now.
        new_pos = self.state["ee_pos"] + delta_pos
        new_ori = self.state["ee_ori"] + delta_ori
        self.robot.update_desired_ee_pos(new_pos, new_ori)


class JointPositionController(Controller):
    @property
    def robot_action_space(self):
        return gym.spaces.Box(low=self.JOINT_LOW, high=self.JOINT_HIGH, dtype=np.float32)

    def start_robot(self):
        self.robot.start_joint_impedance()

    def update_robot(self, action):
        action = np.clip(action, low=self.JOINT_LOW, high=self.JOINT_HIGH)
        self.robot.update_desired_joint_positions(action)


class JointDeltaController(Controller):
    def __init__(self, *args, max_delta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_delta = max_delta

    @property
    def robot_action_space(self):
        high = self._max_delta * np.ones(self.JOINT_LOW.shape[0], dtype=np.float32)
        return gym.spaces.Box(low=-1 * high, high=high, dtype=np.float32)

    def start_robot(self):
        self.robot.start_joint_impedance()

    def update_robot(self, action):
        action = np.clip(action, low=self.robot_action_space.low, high=self.robot_action_space.high)
        self.robot.update_desired_joint_positions(self.state["joint_positions"] + action)


class JointVelocityController(Controller):
    @property
    def action_space(self):
        # Return the action space shape WITH Bounds.
        high = 0.1 * np.ones(self.JOINT_LOW, dtype=np.float32)
        return gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def start_robot(self):
        self.robot.start_joint_veloicty_control()

    def update_robot(self, action):
        self.robot.update_desired_joint_velocities(action)


def precise_wait(t_end: float, slack_time: float = 0.001):
    t_start = time.time()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time.time() < t_end:
            pass
    return


class FrankaEnv(gym.Env):
    """
    A simple Gym Environment for the Franka robots controlled via PolyMetis.
    TODO: support for images etc.
    """

    def __init__(
        self,
        ip_address: str,
        controller: str = "cartesian_velocity",
        control_hz: float = 15.0,
        camera_kwargs: Optional[Dict] = None,
    ):
        self.controller = {
            "cartesian_position": CartesianPositionController,
            "cartesian_delta": CartesianDeltaController,
            "joint_position": JointPositionController,
            "joint_delta": JointDeltaController,
            "joint_velocity": JointVelocityController,
        }[controller](ip_address=ip_address)
        self.action_space = self.controller.action_space
        self.control_hz = float(control_hz)
        self._time = None  # Set time to None.

    def step(self, action):
        # Immediately update with the action
        self.controller.update(action)

        # Wait until we have 15hz since last time.
        precise_wait(self._time + 1 / self.control_hz)
        self._time = time.time()

        state = self.controller.get_state()
        return state, 0, False, {}

    def reset(self):
        self.controller.reset()
        self.cameras.get_obs(0)
        state = self.controller.get_state()
        # start the timer.
        self._time = time.time()
        return state


class FrankaReach(FrankaEnv):
    """
    A simple environment where the goal is for the Franka to reach a specific end effector position.
    """

    def __init__(self, *args, goal_position=(0.5, 0.3, 0.5), **kwargs):
        self._goal_position = np.array(goal_position)
        super().__init__(*args, **kwargs)

    def step(self, action):
        state, reward, done, info = super().step(action)
        reward = -0.1 * np.linalg.norm(state["ee_pos"] - self._goal_position)
        return state, reward, done, info

"""
Simple wrapper for registering metaworld enviornments
properly with gym.
"""
import gym
import numpy as np


class MetaWorldSawyerEnv(gym.Env):
    def __init__(self, env_name, seed=True):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        self._env = ALL_V2_ENVIRONMENTS[env_name]()
        self._env._freeze_rand_vec = False
        self._env._set_task_called = True
        self._seed = seed
        if self._seed:
            self._env.seed(0)  # Seed it at zero for now.

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._max_episode_steps = self._env.max_path_length

    def seed(self, seed=None):
        super().seed(seed=seed)
        if self._seed:
            self._env.seed(0)

    def evaluate_state(self, state, action):
        return self._env.evaluate_state(state, action)

    def step(self, action):
        self._episode_steps += 1
        obs, reward, done, info = self._env.step(action)
        if self._episode_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        # Add the underlying state to the info
        state = self._env.sim.get_state()
        info["state"] = np.concatenate((state.qpos, state.qvel), axis=0)
        return obs.astype(np.float32), reward, done, info

    def set_state(self, state):
        qpos, qvel = state[: self._env.model.nq], state[self._env.model.nq :]
        self._env.set_state(qpos, qvel)

    def reset(self, **kwargs):
        self._episode_steps = 0
        return self._env.reset(**kwargs).astype(np.float32)

    def render(self, mode="rgb_array", camera_name="corner2", width=640, height=480):
        assert mode == "rgb_array", "Only RGB array is supported"
        # stack multiple views
        for ctx in self._env.sim.render_contexts:
            ctx.opengl_context.make_context_current()
        return self._env.render(offscreen=True, camera_name=camera_name, resolution=(width, height))

    def __getattr__(self, name):
        return getattr(self._env, name)


class MetaWorldSawyerImageWrapper(gym.Wrapper):
    def __init__(self, env, width=64, height=64, camera="corner2", show_goal=False):
        assert isinstance(
            env.unwrapped, MetaWorldSawyerEnv
        ), "MetaWorld Wrapper must be used with a MetaWorldSawyerEnv class"
        super().__init__(env)
        self._width = width
        self._height = height
        self._camera = camera
        self._show_goal = show_goal
        shape = (3, self._height, self._width)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def _get_image(self):
        if not self._show_goal:
            try:
                self.env.unwrapped._set_pos_site("goal", np.inf * self.env.unwrapped._target_pos)
            except ValueError:
                pass  # If we don't have the goal site, just continue.
        img = self.env.render(mode="rgb_array", camera_name=self._camera, width=self._width, height=self._height)
        return img.transpose(2, 0, 1)

    def get_state_obs(self):
        return self.env.unwrapped._get_obs()

    def step(self, action):
        state_obs, reward, done, info = self.env.step(action)
        # Throw away the state-based observation.
        info["state_obs"] = state_obs
        return self._get_image().copy(), reward, done, info

    def reset(self):
        # Zoom in camera corner2 to make it better for control
        # I found this view to work well across a lot of the tasks.
        camera_name = "corner2"
        # Original XYZ is 1.3 -0.2 1.1
        index = self.model.camera_name2id(camera_name)
        self.model.cam_fovy[index] = 20.0  # FOV
        self.model.cam_pos[index][0] = 1.5  # X
        self.model.cam_pos[index][1] = -0.35  # Y
        self.model.cam_pos[index][2] = 1.1  # Z

        self.env.reset()
        return self._get_image().copy()  # Return the image observation


def get_mw_image_env(env_name):
    env = MetaWorldSawyerEnv(env_name)
    return MetaWorldSawyerImageWrapper(env)

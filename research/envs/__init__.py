# If we want to register environments in gym.
# These will be loaded when we import the research package.
from gym.envs import register

from .base import EmptyEnv

# Add things you explicitly want exported here.
# Otherwise, all imports are deleted.
__all__ = ["EmptyEnv"]

try:
    import gym_robotics
except ImportError:
    print("[research] skipping gym robotics, package not found.")

try:
    import d4rl
except ImportError:
    print("[research] skipping d4rl, package not found.")

try:
    # Register environment classes here
    # Register the DM Control environments.
    from dm_control import suite

    # Custom DM Control domains can be registered as follows:
    # from . import <custom dm_env module>
    # assert hasattr(<custom dm_env module>, 'SUITE')
    # suite._DOMAINS['<custom dm_env module>'] = <custom dm_env module>

    # Register all of the DM control tasks
    for domain_name, task_name in suite._get_tasks(tag=None):
        # Import state domains
        ID = f"{domain_name.capitalize()}{task_name.capitalize()}-v0"
        register(
            id=ID,
            entry_point="research.envs.dm_control:DMControlEnv",
            kwargs={
                "domain_name": domain_name,
                "task_name": task_name,
                "action_minimum": -1.0,
                "action_maximum": 1.0,
                "action_repeat": 1,
                "from_pixels": False,
                "flatten": True,
                "stack": 1,
            },
        )

        # Import vision domains as specified in DRQ-v2
        ID = f"{domain_name.capitalize()}{task_name.capitalize()}-vision-v0"
        camera_id = dict(quadruped=2).get(domain_name, 0)
        register(
            id=ID,
            entry_point="research.envs.dm_control:DMControlEnv",
            kwargs={
                "domain_name": domain_name,
                "task_name": task_name,
                "action_repeat": 2,
                "action_minimum": -1.0,
                "action_maximum": 1.0,
                "from_pixels": True,
                "height": 84,
                "width": 84,
                "camera_id": camera_id,
                "flatten": False,
                "stack": 3,
            },
        )

    # Cleanup extra imports
    del suite
except ImportError:
    print("[research] Skipping dm_control, package not found.")

try:
    from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

    # Add the meta world test environments.
    # For each one, register the different tasks.

    for env_name, _env_cls in ALL_V2_ENVIRONMENTS.items():
        ID = f"mw_{env_name}"
        register(id=ID, entry_point="research.envs.metaworld:MetaWorldSawyerEnv", kwargs={"env_name": env_name})
        id_parts = ID.split("-")
        id_parts[-1] = "image-" + id_parts[-1]
        ID = "-".join(id_parts)
        register(id=ID, entry_point="research.envs.metaworld:get_mw_image_env", kwargs={"env_name": env_name})
except ImportError:
    print("[research] Skipping metaworld, package not found.")

try:
    from .robomimic import RobomimicEnv
except ImportError:
    print("[research] Skipping robomimic, package not found")

try:
    from .franka import FrankaEnv, FrankaReach
except ImportError:
    print("[research] Skipping polymetis, package not found.")

del register

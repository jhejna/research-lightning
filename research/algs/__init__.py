# Register Algorithms here.

from .offline.bc import BehaviorCloning
from .offline.dp import DiffusionPolicy
from .offline.iql import IQL
from .online.dqn import DQN, DoubleDQN, SoftDoubleDQN, SoftDQN
from .online.drqv2 import DRQV2
from .online.ppo import PPO, AdaptiveKLPPO
from .online.sac import SAC
from .online.td3 import TD3

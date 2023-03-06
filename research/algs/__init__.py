# Register Algorithms here.

# Online Algorithms
from .online.td3 import TD3
from .online.sac import SAC
from .online.drqv2 import DRQV2
from .online.ppo import PPO, AdaptiveKLPPO
from .online.dqn import DQN, DoubleDQN, SoftDQN, SoftDoubleDQN

# Offline Algorithms
from .offline.iql import IQL
from .offline.bc import BehaviorCloning

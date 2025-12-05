from algorithm.imitation import Agent as Imitation
from algorithm.ppo import Agent as PPO
from algorithm.sac import Agent as SAC

algo_dict = {
    'imitation': Imitation,
    'ppo': PPO,
    'sac': SAC,
    }
from gym.envs.registration import register

register(
    id='vrep_soccer-v0',
    entry_point='vrep_env.envs:VrepSoccerEnv'
)
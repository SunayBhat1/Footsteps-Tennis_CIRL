from gym.envs.registration import register

register(
    id='PaperTennis-v0',
    entry_point='PaperTennis.envs:PaperTennisEnv',
    max_episode_steps = 51,
)
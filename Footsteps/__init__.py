from gym.envs.registration import register


register(
    id='Footsteps-v0',
    entry_point='Footsteps.envs:FootstepsEnv',
    max_episode_steps = 51,
)
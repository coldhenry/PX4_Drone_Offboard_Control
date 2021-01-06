from gym.envs.registration import register

register(
        id = 'NLinkArm-v0',
        entry_point = 'custom_gym.envs.NLinkArm:ArmPointEnv',
        )

register(
        id = 'QuadRotor-v0',
        entry_point = 'custom_gym.envs.QuadRotor:QuadRotorEnv'
        )
register(
        id = 'Quad-v1',
        entry_point = 'custom_gym.envs.Quad:QuadRotorEnv'
        )
register(
        id = 'Humanoid-v0',
        entry_point = 'custom_gym.envs.Humanoid:HumanoidEnv'
        )
register(
        id = 'Stanley-v0',
        entry_point = 'custom_gym.envs.Stanley:Stanley_Env'
        )

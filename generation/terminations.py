"""每个环境 step 是一次完整尝试，正常 terminal，不跨尝试 bootstrap。"""


def attempt_complete(env):
    return env.episode_length_buf >= 1

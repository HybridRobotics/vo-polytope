class env_obs_line:
    def __init__(self, obs_line_states=None, **kwargs):
        if obs_line_states is None:
            obs_line_states = []
        self.obs_line_states = obs_line_states

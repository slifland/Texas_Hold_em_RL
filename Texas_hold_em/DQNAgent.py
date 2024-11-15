class DQNAgent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = True

    def eval_step(self, state):
        state = state['raw_obs']
        legal_actions = state['legal_actions']
        return 'fold'

    def step(self, state):
        state = state['raw_obs']
        legal_actions = state['legal_actions']
        return 'fold'

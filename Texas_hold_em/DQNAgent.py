from torch import nn
from collections import deque
import torch.nn.functional as F
import random

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)  # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions)  # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply rectified linear unit (ReLU) activation
        x = self.out(x)  # Calculate output
        return x


# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = True

    def eval_step(self, state):
        state = state['raw_obs']
        legal_actions = state['legal_actions']
        return 'fold', {}

    def step(self, state):
        state = state['raw_obs']
        legal_actions = state['legal_actions']
        return 'fold'

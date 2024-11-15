from torch import nn
import torch
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

    def state_to_dqn_input(self, state : dict[str, int]) -> torch.Tensor:
        input_tensor = torch.zeros(52)
        #0-12 are spade A - K
        #13-25 are heart A - K
        #26-38 are diamond A - K
        #39-51 club A - K
        # 0 represents not seen, 1 represents in hand, 2 represents public card
        for value in state['hand'].values():
            val = self.determineIndex(value)
            input_tensor[val] = 1
        for value in state['public_cards'].values():
            val = self.determineIndex(value)
            input_tensor[val] = 2
        return input_tensor
    def determineIndex(self, card):
        val = 0
        match card[0]:
            case 'S':
                val = 0
            case 'H':
                val = 13
            case 'D':
                val = 26
            case 'C':
                val = 39
        match card[1]:
            case 'A':
                val += 0
            case 'T':
                val += 10
            case 'Q':
                val += 11
            case 'K':
                val += 12
            case _:
                val += card[1]
        return val

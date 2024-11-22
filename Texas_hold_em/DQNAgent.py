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
        self.discount_factor_g = 0.9
        self.num_actions = num_actions
        self.use_raw = True
        self.policy_dqn = DQN(57, 57, num_actions)
        self.target_dqn = DQN(57, 57, num_actions)
        self.replay_memory = ReplayMemory(20000)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=1e-3)
        self.epsilon = 0.1
        self.step_count = 0

    def action_as_string(self, action):
        match action:
            case 0:
                return 'call'
            case 1:
                return 'raise'
            case 2:
                return 'fold'
            case 3:
                return 'check'

    def eval_step(self, state):
        state = state['raw_obs']
        legal_actions = state['legal_actions']
        # Select action based on epsilon-greedy
        if random.random() < self.epsilon:
            # select random action
            action = random.choice(legal_actions)
        else:
            # select best action
            with torch.no_grad():
                action = self.policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                action = self.action_as_string(action)
        # Decay epsilon
        self.epsilon = max(self.epsilon - 0.0001, 0)
        if action in legal_actions:
            return action, {}
        return 'fold', {}

    def step(self, state):
        state = state['raw_obs']
        legal_actions = state['legal_actions']
        # Select action based on epsilon-greedy
        if random.random() < self.epsilon:
            # select random action
            action = random.choice(legal_actions)
        else:
            # select best action
            with torch.no_grad():
                action = self.policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                action = self.action_as_string(action)
        # Decay epsilon
        self.epsilon = max(self.epsilon - 0.0001, 0)
        if action in legal_actions:
            return action
        return 'fold'

    def state_to_dqn_input(self, state : {}) -> torch.Tensor:
        input_tensor = torch.zeros(57)
        #0-12 are spade A - K
        #13-25 are heart A - K
        #26-38 are diamond A - K
        #39-51 club A - K
        # 0 represents not seen, 1 represents in hand, 2 represents public card
        if 'hand' in state:
            for value in state['hand']:
                val = self.determineIndex(value)
                input_tensor[val] = 1
        if 'public_cards' in state:
            for value in state['public_cards']:
                val = self.determineIndex(value)
                input_tensor[val] = 2
        #number of chips you have
        if('my_chips' in state):
            input_tensor[52] = state['my_chips']
        #0 for invalid action, 1 for valid action - call, raise, fold, check
        input_tensor[53] = 1 if ('call'in state['legal_actions']) else 0
        input_tensor[54] = 1 if ('raise' in state['legal_actions']) else 0
        input_tensor[55] = 1 if ('fold' in state['legal_actions']) else 0
        input_tensor[56] = 1 if ('check' in state['legal_actions']) else 0
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
                val += 1
            case 'T':
                val += 10
            case 'J':
                val += 11
            case 'Q':
                val += 12
            case 'K':
                val += 13
            case _:
                val += int(card[1])
        return val - 1

    def save_to_memory(self, transition):
        self.replay_memory.append(transition)

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        ts[0] = ts[0]['raw_obs']
        self.save_to_memory((state['raw_obs'], action, reward, next_state['obs'], list(next_state['legal_actions'].keys()), done))
        self.train()
        pass

    def train(self):
        current_q_list = []
        target_q_list = []
        to_train = self.replay_memory.sample(min(len(self.replay_memory), 1000))
        for state, action, reward, next_state, done, legal_actions in to_train:
            if done:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * self.target_dqn(
                            self.state_to_dqn_input(next_state).max())
                    )
            current_q = self.policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            target_q = self.target_dqn(self.state_to_dqn_input(state))
            #adjust line - assumes action is coded as int, but its coded as string here
            action = self.action_string_to_int(action)
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        if(self.step_count % 100 == 0):
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            self.step_count = 0

    def action_string_to_int(self, action):
        match action:
            case 'fold':
                return 0
            case 'call':
                return 1
            case 'raise':
                return 2
            case 'check':
                return 3




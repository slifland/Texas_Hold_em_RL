import random


class ConservativePokerAgent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = True

    def eval_step(self, state):
        state = state['raw_obs']
        #print(state)
        legal_actions = state['legal_actions']
        hand_strength = self.evaluate_hand(state)
        if hand_strength >= 0.8:
            if('raise' in legal_actions):
                return "raise", {}
            return 'call', {}
        elif hand_strength >= 0.6:
            if('check' in legal_actions):
                return "check", {}
            return "call", {}
        else:
            if('check' in legal_actions):
                return "check", {}
            return "fold", {}

        
    def step(self, state):
        state = state['raw_obs']
        legal_actions = state['legal_actions']
        hand_strength = self.evaluate_hand(state)
        fold, call, raise_ = 2, 0, 1
        if hand_strength >= 0.8:
            if('raise' in legal_actions):
                return "raise"
            return 'call'
        elif hand_strength >= 0.6:
            if('check' in legal_actions):
                return "check"
            return "call"
        else:
            if('check' in legal_actions):
                return "check"
            return "fold"

    def evaluate_hand(self, state):
        cards = state['hand'] + state['public_cards']
        strength = self.simple_hand_strength(cards)
        return strength

    def simple_hand_strength(self, cards):
        ranks = [card[1] for card in cards]
        rank_counts = {rank: ranks.count(rank) for rank in ranks}
        if max(rank_counts.values()) >= 2:
            return 0.8  
        if any(rank in ['A', 'K', 'Q', 'J'] for rank in ranks):
            return 0.6  
        return 0.4  


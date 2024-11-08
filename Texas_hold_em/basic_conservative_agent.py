import random


class ConservativePokerAgent():
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def eval_step(self, state):
        hand_strength = self.evaluate_hand(state)
        fold, call, raise_ = 0, 1, 2

        if hand_strength >= 0.8:
            return raise_
        elif hand_strength >= 0.6:
            return call
        else:
            return fold

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

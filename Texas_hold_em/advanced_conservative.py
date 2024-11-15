from collections import Counter

class AdvancedPokerAgent():
    def card_rank(card):
        rank = card[1]
        if rank == 'T':
            return 10
        elif rank == 'J':
            return 11
        elif rank == 'Q':
            return 12
        elif rank == 'K':
            return 13
        elif rank == 'A':
            return 14
        else:
            return int(rank)

    def card_suit(card):
        return card[0]

    def is_flush(cards):
        suits = [card_suit(card) for card in cards]
        return len(set(suits)) == 1

    def is_straight(ranks):
        sorted_ranks = sorted(ranks)
        return sorted_ranks == list(range(min(ranks), max(ranks) + 1))

    def classify_hand(cards):
        ranks = [card_rank(card) for card in cards]
        rank_counts = Counter(ranks)
        distinct_ranks = len(rank_counts)
        highest_count = max(rank_counts.values())
        flush = is_flush(cards)
        straight = is_straight(ranks)
        max_rank = max(ranks)
        
        if flush and straight and max_rank == 14:  
            return 10
        elif flush and straight:  
            return 9
        elif highest_count == 4:  
            return 8
        elif highest_count == 3 and distinct_ranks == 2:  
            return 7
        elif flush:  
            return 6
        elif straight: 
            return 5
        elif highest_count == 3: 
            return 4
        elif highest_count == 2 and distinct_ranks == 3: 
            return 3
        elif highest_count == 2:  
            return 2
        else: 
            return 1

    def poker_hand_heuristic(cards):
        base_score = classify_hand(cards)
        ranks = [card_rank(card) for card in cards]
        rank_counts = Counter(ranks)
        
        sorted_ranks = sorted(rank_counts.items(), key=lambda x: (-x[1], -x[0]))
        score = base_score * 1000000
        
        multiplier = 100000  
        for rank, count in sorted_ranks:
            score += rank * multiplier * count
            multiplier //= 10
        
        return score
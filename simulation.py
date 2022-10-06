import numpy as np

num_players = 6
buyin = 200
sb = 1
bb = 2
num_hands = 10000

class Player():
    def __init__(self):
        self.fold_mean = np.random.rand()
        self.fold_std = np.random.rand()
        self.raise_mean = np.random.rand()
        self.raise_std = np.random.rand()
        self.chips = buyin
        self.history = np.zeros((num_hands, 4))
        self.hands_played = 0
        self.start_hand = 0

def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum()

def play_hand(players, button, hand):
    start_chips = [p.chips for p in players]
    cards = np.random.rand(num_players)
    bets = np.zeros(num_players)
    fold = [False] * num_players

    # sb
    sb_idx = (button + 1) % num_players
    sb_bet = min(sb, players[sb_idx].chips)
    players[sb_idx].chips -= sb_bet
    bets[sb_idx] += sb_bet

    # bb
    bb_idx = (button + 1) % num_players
    bb_bet = min(bb, players[bb_idx].chips)
    players[bb_idx].chips -= bb_bet
    bets[bb_idx] += bb_bet

    # action
    idx = (button + 3) % num_players
    total_bet = bb
    while any(p.chips > 0 for i, p in enumerate(players) if not fold[i]):
        player = players[idx]
        if not fold[idx] and player.chips > 0:
            diff = total_bet - bets[idx]
            if diff > 0:
                fold_val = np.random.randn() * player.fold_std + player.fold_mean
                if fold_val <= cards[idx]:
                    # call or raise
                    player.history[hand, 2] = 1
                    raise_val = np.random.randn() * player.raise_std + player.raise_mean
                    if raise_val >= cards[idx]:
                        # raise
                        player.history[hand, 3] = 1
                        amount = min(2 * diff, player.chips)
                        player.chips -= amount
                        bets[idx] += amount
                        increase = max(amount - diff, 0)
                        total_bet += increase
                    else:
                        # call
                        amount = min(diff, player.chips)
                        player.chips -= amount
                        bets[idx] += amount
                else:
                    # fold
                    fold[idx] = True
            else:
                # action finished (option always checks)
                break
        idx = (idx + 1) % num_players

    # determine winner
    pot = bets.sum()
    players_in = [p for i, p in enumerate(players) if not fold[i]]
    cards_in = np.array([c for i, c in enumerate(cards) if not fold[i]])
    win_probs = softmax(np.log(cards_in))
    winner = np.random.choice(players_in, p=win_probs)
    winner.chips += pot

    # update stats
    for i, p in enumerate(players):
        p.history[hand, 0] = p.chips
        p.history[hand, 1] = p.chips - start_chips[i]
        p.hands_played += 1

def simulate():
    players = [Player() for i in range(num_players)]
    eliminated = []
    for hand in range(num_hands):
        button = hand % num_players
        play_hand(players, button, hand)
        # replace eliminated players
        eliminated.extend([p for p in players if p.chips == 0])
        players = [p for p in players if p.chips > 0]
        diff = num_players - len(players)
        for i in range(diff):
            new = Player()
            new.start_hand = hand + 1
            players.append(new)
    return players, eliminated

if __name__ == "__main__":
    players, eliminated = simulate()

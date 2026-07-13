import numpy as np
import matplotlib.pyplot as plt


starting_bankroll = 100
win_prob = 0.51
bet_size = 1

n_bets = 1000

def simulate_bets(starting_bankroll, win_prob, bet_size, n_bets):
    if starting_bankroll <= 0:
        return 0
    
    choices = np.random.random(size=n_bets)
    wins = choices < win_prob
    payoffs = np.where(wins, bet_size, -bet_size)
    current_bankroll = np.cumsum(payoffs) + starting_bankroll
    return np.any(current_bankroll < 0)


def mc_simulate(starting_bankroll, win_prob, bet_size, n_bets, n_sims):
    nsuccess = 0
    for _ in range(n_sims):
        out = simulate_bets(starting_bankroll, win_prob, bet_size, n_bets)
        if out > 0:
            nsuccess += 1
    return nsuccess / n_sims


bets_list = []
results_list = []
for n_bets in [100, 500, 1000, 2000, 3000, 4000, 5000, 10000]:
    result = mc_simulate(starting_bankroll, win_prob, bet_size, n_bets, 10000)
    bets_list.append(n_bets)
    results_list.append(result)

print("Hello goi gn to plot")

plt.scatter(bets_list, results_list)
plt.xlabel("Number of bets")
plt.ylabel("P(ruin)")
plt.show()
print("After")

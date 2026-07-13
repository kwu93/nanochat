# Experiment Journal

A running log of side experiments and explorations done alongside nanochat work, plus what each one taught me.
Newest entries first.

## 2026-07-13 - Gambler's ruin under a small positive edge

**Files:** `gamblers.py`, `gamblers.ipynb`

**Question:** If I bet with a slight edge (51% win probability, even money, flat bet size), what is the probability of ruin, and how does it change as I keep playing more rounds?

**Setup:**
- Starting bankroll: 100 units.
- Each bet wins +1 with probability 0.51 and loses -1 with probability 0.49 (positive expected value per bet).
- Flat bet size of 1 unit, no bet sizing or Kelly scaling.
- "Ruin" is defined as the bankroll dipping below 0 at any point during the sequence, not just at the end.
- Estimated via Monte Carlo with 10,000 simulations per horizon.

**Results (P(ruin) vs number of bets):**

| Number of bets | P(ruin) |
|----------------|---------|
| 100            | 0.0000  |
| 500            | 0.0000  |
| 1,000          | 0.0001  |
| 2,000          | 0.0019  |
| 3,000          | 0.0052  |
| 4,000          | 0.0098  |
| 5,000          | 0.0116  |
| 10,000         | 0.0148  |

**Learnings:**
- Even with a favorable edge, the probability of ruin is not zero and grows with the number of bets, because a long-enough game gives more chances for an early losing streak to wipe out the bankroll before the edge compounds.
- The growth is not unbounded, though: it rises quickly at first and then flattens out.
  This matches the classic gambler's ruin result that with a positive edge the probability of *ever* being ruined converges to a finite limit below 1, rather than approaching certainty.
- For this configuration the theoretical asymptotic ruin probability is `(q/p)^bankroll = (0.49/0.51)^100 ≈ 0.0183`.
  The Monte Carlo estimate at 10,000 bets (0.0148) is climbing toward that ceiling, which is a nice sanity check that the simulation is behaving correctly.
- Practical takeaway: a small edge plus a decent bankroll makes ruin fairly unlikely, but "unlikely" is horizon-dependent, and the flat-betting version leaves a persistent tail risk that only bet sizing (e.g. Kelly) would reduce.

## 2026-07-13 - Peeking at the ClimbMix base training data

**Files:** `explore.ipynb`

**Question:** What do the ClimbMix pretraining shards actually look like on disk, so I can reason about the base data pipeline?

**Setup:**
- Tried to `pd.read_parquet` a shard directly from the nanochat cache at `~/.cache/nanochat/base_data_climbmix/shard_0001.parquet`.

**Learnings:**
- The shards are not present in the local cache yet, so the read failed with `FileNotFoundError`.
- This is expected: the ClimbMix data has to be downloaded first (the training scripts pull it on demand), and it was never fetched on this machine.
- Follow-up if I want to actually inspect the data: run the dataset download step, or point the notebook at a shard that has already been materialized, before reloading it into pandas.

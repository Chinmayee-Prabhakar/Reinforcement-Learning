# Reinforcement Learning — From Bandits to Dyna-style Planning

Five Python projects covering exploration, dynamic programming, Monte Carlo estimation, temporal-difference control, and the integration of learning with planning.

## The five projects

| Project | Algorithms and ideas | Source | Report |
| --- | --- | --- | --- |
| 1 | 10-armed bandit, greedy and ε-greedy action selection | [`Bandit.py`](Project%201/Bandit.py) | [PDF](Project%201/Project%20Report.pdf) |
| 2 | Value iteration, policy iteration, generalized policy iteration | [`dynamic_programming.py`](Project%202/dynamic_programming.py) | [PDF](Project%202/PA2_RL_Report.pdf) |
| 3 | Monte Carlo first-visit and exploring-starts exercises | [`montecarlo_main.py`](Project%203/montecarlo_main.py) | [PDF](Project%203/PA3_RL_Report.pdf) |
| 4 | SARSA and off-policy Q-learning on a cliff-walking task | [`PA4_TD_finalcode.py`](Project%204/PA4_TD_finalcode.py) | [PDF](Project%204/PA4_TD_report.pdf) |
| 5 | Dyna-style maze planning with configurable planning steps | [`dyna_maze.py`](Project%205/dyna_maze.py) | [PDF](Project%205/RL_PA5_report.pdf) |

```mermaid
flowchart LR
 A[Exploration in bandits] --> B[Model-based DP]
 B --> C[Monte Carlo returns]
 C --> D[TD learning: SARSA / Q-learning]
 D --> E[Dyna-style planning updates]
```

### Project highlights

- **Bandits:** the code explores ε values 0, 0.01 and 0.1 across simulated action-reward problems.
- **Dynamic programming:** derives value functions and greedy policies when the model is known.
- **Monte Carlo:** estimates action values from complete episodes.
- **Cliff walking:** illustrates different on-policy and off-policy temporal-difference updates.
- **Dyna maze:** compares learning curves under different simulated planning-step budgets.

For plots and original conclusions, open the corresponding reports.

## Reproduction

Each project is a standalone script and should run from its own directory, for example:

```bash
cd "Project 4"
python PA4_TD_finalcode.py
```

NumPy and Matplotlib are used throughout.


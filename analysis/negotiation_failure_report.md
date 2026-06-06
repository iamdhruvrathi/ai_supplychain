# Negotiation Failure Report

## Data Source
- Experiment: exp3_negotiation
- Runs: 30
- Trajectory file: results/exp3_negotiation/trajectories/rollouts.csv
- Metrics: total cost, backlog, inventory, consensus gap

## Top 5 Worst Runs
- Run 1: total cost=12168.0, max backlog=276, max order=56, inventory collapse events=41, backlog explosion events=18, mean gap=0.00, max gap=0.0
- Run 16: total cost=11147.0, max backlog=234, max order=56, inventory collapse events=49, backlog explosion events=22, mean gap=2.50, max gap=16.0
- Run 14: total cost=10184.0, max backlog=236, max order=56, inventory collapse events=40, backlog explosion events=17, mean gap=2.00, max gap=40.0
- Run 4: total cost=9164.0, max backlog=216, max order=56, inventory collapse events=47, backlog explosion events=18, mean gap=3.40, max gap=16.0
- Run 13: total cost=7164.0, max backlog=200, max order=56, inventory collapse events=43, backlog explosion events=11, mean gap=2.20, max gap=20.0

## Top 5 Best Runs
- Run 0: total cost=560.0, max backlog=12, max order=16, inventory collapse events=18, backlog explosion events=0, mean gap=0.20, max gap=4.0
- Run 10: total cost=560.0, max backlog=12, max order=16, inventory collapse events=18, backlog explosion events=0, mean gap=0.20, max gap=4.0
- Run 18: total cost=560.0, max backlog=12, max order=16, inventory collapse events=18, backlog explosion events=0, mean gap=0.20, max gap=4.0
- Run 25: total cost=568.0, max backlog=12, max order=16, inventory collapse events=28, backlog explosion events=0, mean gap=0.20, max gap=4.0
- Run 2: total cost=570.0, max backlog=12, max order=16, inventory collapse events=16, backlog explosion events=0, mean gap=0.00, max gap=0.0

## Statistical Comparison
| Metric | Worst runs mean | Best runs mean |
|---|---|---|
| Max backlog | 232.4 | 12.0 |
| Max order | 56.0 | 16.0 |
| Inventory collapse events | 44.0 | 19.6 |
| Backlog explosion events | 17.2 | 0.0 |
| Mean consensus gap | 2.0 | 0.2 |

## Hypothesis for Failure Mechanism
Worst negotiation runs are distinguished by severe backlog growth and repeated inventory depletion. High-cost runs show sustained backlog explosion events (backlog >= 100) while their maximum order remains capped at 56, suggesting negotiation timing and coordination delays prevent sufficient replenishment. Best runs maintain low backlog, low order variance, and near-zero consensus gap, implying smooth agreement and fast correction.

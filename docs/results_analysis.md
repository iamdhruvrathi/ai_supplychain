# Results Analysis

## Experimental Setup

- Model: `Qwen/Qwen3-4B`
- Backend: `vLLM`
- Runs per condition: `30`
- Simulation length: `20` weeks
- Demand: fixed Beer Game demand path
- Conditions:
  - `Baseline`
  - `Tool`
  - `Negotiation`
  - `Tool+Negotiation`

## Result Table

| Condition | Mean Cost | CV | Ψ | Φ | Consensus Gap |
|---|---|---|---|---|---|
| Baseline | 16491.5 | 0.16 | 1.38 | 1.68 | 18.11 |
| Tool | 15281.5 | 0.04 | 1.88 | 1.01 | 29.03 |
| Negotiation | 2804.0 | 1.25 | 1.03 | 1.29 | 1.26 |
| Tool+Negotiation | 12851.5 | 0.05 | 2.12 | 1.28 | 17.79 |

## Observations

- `Negotiation` is the strongest driver of cost reduction and consensus improvement.
- `Tool` alone reduces mean cost relative to the baseline in this reported table, but increases consensus gap substantially.
- `Tool+Negotiation` reduces mean cost versus baseline, but still has notable consensus gap compared to pure negotiation.

## Figure References

- `plots/research/cost_distribution.png`: Total cost distributions by condition
- `plots/research/consensus_gap_over_time.png`: Consensus gap trajectories across weeks
- `plots/research/reliability_tradeoff.png`: CV vs mean cost tradeoff

## Negotiation Findings

- Worst negotiation runs show severe backlog growth and repeated inventory depletion.
- Best negotiation runs maintain low backlog, low maximum order, and near-zero consensus gap.
- The main failure mechanism appears to be coordination delay: backlog grows faster than orders can correct, and consensus remains weak in bad runs.

## Reliability Findings

- `Negotiation` has the lowest mean cost but appears to have the highest CV in the table.
- `Tool` and `Tool+Negotiation` show low CV but mixed performance, suggesting steady yet potentially biased behavior.
- The reliability tradeoff plot highlights how these conditions balance cost versus variability.

## Tool Findings

- Tool guidance appears to alter order distributions and consensus dynamics.
- When combined with negotiation, the tool still raises consensus gap relative to pure negotiation.
- Further research should isolate whether the tool is introducing systematic order bias or simply changing the equilibrium behavior.

## Notes

- This analysis reuses existing experiment outputs in `results/exp1_baseline`, `results/exp2_tool`, `results/exp3_negotiation`, and `results/exp4_tool_negotiation`.
- Figures were generated with new analysis scripts under `analysis/` and saved to `plots/research/`.

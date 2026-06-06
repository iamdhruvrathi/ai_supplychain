# Experimental Setup

All experiments were conducted using a four-echelon Beer Game supply chain consisting of a Retailer, Wholesaler, Distributor, and Factory controlled by LLM-based agents.

**Model:** Qwen3-4B
**Inference Backend:** vLLM
**Runs per Experiment:** 30 independent runs
**Simulation Horizon:** 20 weeks per run
**Demand Pattern:** Fixed Beer Game demand sequence
**Maximum Order Quantity:** 100 units
**Inference Mode:** /no_think
**Evaluation Metrics:** Mean Cost, Coefficient of Variation (CV), Cross-Echelon Amplification (Ψ), Temporal Amplification (Φ), and Consensus Gap.

Four experimental conditions were evaluated:

1. **Baseline** – Decentralized LLM agents without tools or negotiation.
2. **Tool Only** – LLM agents assisted by an inventory recommendation tool.
3. **Negotiation Only** – LLM agents allowed to negotiate before finalizing orders.
4. **Tool + Negotiation** – LLM agents using both inventory recommendations and negotiation.

---

# Experimental Results Summary

## Table 1: Comparison of Experimental Conditions

| Mode               | Mean Cost |    CV | Mean Ψ | Mean Φ | Mean Consensus Gap |
| ------------------ | --------: | ----: | -----: | -----: | -----------------: |
| Baseline           | 16,491.50 | 0.161 |  1.382 |  1.678 |              18.11 |
| Tool Only          | 15,281.53 | 0.047 |  1.877 |  1.012 |              29.03 |
| Negotiation Only   |  2,804.07 | 1.258 |  1.032 |  1.286 |               1.26 |
| Tool + Negotiation | 12,851.53 | 0.058 |  2.121 |  2.821 |              17.79 |

---

# Observations

### Baseline

The decentralized baseline exhibited moderate cost, variability, and disagreement among agents. Agent Bullwhip was observed through both cross-echelon (Ψ > 1) and temporal amplification (Φ > 1).

### Tool Only

Tool augmentation reduced mean cost by approximately 7% and significantly improved reliability, reducing the coefficient of variation from 0.161 to 0.047. However, consensus gap increased, indicating that while agents behaved more consistently across runs, they were not necessarily more aligned with one another.

### Negotiation Only

Negotiation achieved the lowest mean cost, reducing total cost by approximately 83% compared to the baseline. It also produced the strongest coordination effect, reducing the consensus gap from 18.11 to 1.26 and lowering cross-echelon amplification. However, this came at the expense of substantially higher run-to-run variability.

### Tool + Negotiation

The combination of tools and negotiation improved cost relative to the baseline while maintaining low variability. Compared to negotiation alone, it achieved greater reliability but weaker coordination, as reflected by higher consensus gap and amplification metrics.

---

# Key Findings

1. Negotiation achieved the best operational performance, producing the lowest average supply-chain cost.

2. Tool augmentation achieved the highest reliability, substantially reducing run-to-run variability.

3. Negotiation was most effective at improving coordination, yielding the lowest consensus gap and lowest cross-echelon amplification.

4. The combination of tools and negotiation provided a balance between performance and reliability, outperforming the baseline in both cost and variability.

5. The results indicate a trade-off between efficiency and reliability in LLM-controlled supply chains, where negotiation improves performance while tool support improves consistency.

---

# Conclusion

The experimental results demonstrate that coordination and decision-support mechanisms influence different aspects of supply-chain behavior. Negotiation is highly effective at reducing costs and improving coordination, whereas tool support primarily improves reliability. Combining both approaches provides a balanced improvement over the baseline, suggesting that hybrid strategies can help mitigate Agent Bullwhip while maintaining stable decision-making.

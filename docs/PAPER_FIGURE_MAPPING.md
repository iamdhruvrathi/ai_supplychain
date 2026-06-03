# Beer Game Paper Replication Progress

## Replication Status

| Paper Figure | Repo Script            | Status                      |
| ------------ | ---------------------- | --------------------------- |
| Figure 1     | ?                      | Partial                     |
| Figure 2     | `run_figure2.py`       | ✅ Reproduced (Qualitative) |
| Figure 3     | `run_majority_vote.py` | ⏳ Running                  |
| Figure 4     | Missing                | ❌                          |
| Figure 5     | Missing                | ❌                          |

---

## Experimental Results

| Experiment | Weeks | Runs |     Mean Cost |         CV | Ψ (Mean) | Φ (Mean) |
| ---------- | ----: | ---: | ------------: | ---------: | -------: | -------: |
| N=1        |    20 |   30 | **18,799.03** |  **8.86%** | **2.44** | **4.44** |
| N=10       |    20 |   10 | **17,298.80** | **10.80%** |  Pending |  Pending |
| N=100      |    20 |   10 |       Running |    Running |  Running |  Running |

| Mode               | Weeks | Runs | Mean Cost |         CV |
| ------------------ | ----: | ---: | --------: | ---------: |
| Baseline           |     5 |    2 |     828.5 | **0.5872** |
| Tool               |     5 |    2 |     322.0 |     0.1801 |
| Negotiation        |     5 |    2 |     373.5 |     0.1031 |
| Tool + Negotiation |     5 |    2 | **283.0** | **0.0177** |

### Estimated Overall Progress

**~60–65% paper replication completed**

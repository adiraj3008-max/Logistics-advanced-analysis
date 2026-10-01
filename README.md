# Advanced Data Analysis and Visualization in Logistics

Yuva Intern - Logistics Data Analyst Intern - Week 3 task.

Exploratory data analysis of a **simulated logistics dataset** (5,000 shipments, 5 hubs, 4 carriers,
3 transport modes) using Python: data simulation, cleaning, descriptive statistics, nine visualizations,
statistical tests, driver (regression) analysis and what-if scenarios.

## Key findings
- 24.6% of shipments are delivered late; mean delivery time is 42.0 h, mean cost is $197.74 per shipment.
- One carrier (TransGo) causes about 45% of delays; the Mumbai hub causes about 44% of delay hours.
- Delay rate rises from 13.7% (clear) to 88.1% (storm); Oct-Dec peak season adds about 11 points of delay.
- Distance and transport mode drive time and cost; customer rating tracks delay vs promise, not raw duration.

## Project structure
| Path | Content |
|---|---|
| `logistics_eda.py` | Complete analysis script |
| `data/logistics_shipments_raw.csv` | Simulated data with injected quality issues |
| `data/logistics_shipments_clean.csv` | Cleaned data used for analysis |
| `figures/` | The nine charts used in the report |
| `results.json` | All computed statistics |

## Run
```bash
pip install numpy pandas matplotlib seaborn scipy
python logistics_eda.py
```
The random seed is fixed (42), so results are reproducible.

## Note
The dataset is hypothetical (as required by the task). Effects such as carrier and weather impact were built
into the simulation on purpose, so results demonstrate the method and are not real-world benchmarks.

# `llm-energy-sim` Documentation

## What This Project Does
`llm-energy-sim` is a simple Python simulator that estimates how much energy an LLM server uses when processing user requests. 

Instead of treating inference as one big step, it breaks each request into two phases:
1. **Prefill Phase (Prompt Reading):** Heavy matrix math. High power usage ($P_{\text{prefill}} \approx 250\text{W}$).
2. **Decode Phase (Word Generation):** Token-by-token generation waiting on memory access. Lower power usage ($P_{\text{decode}} \approx 135\text{W}$).

---

## Math & Formulas

For every request $i$:
* $\text{Prefill Time} = \text{prompt\_tokens} \times 0.00005\text{ seconds}$
* $\text{Decode Time} = \text{decode\_tokens} \times 0.002\text{ seconds}$

**Energy Calculation:**
$$\text{Energy (Joules)} = (P_{\text{prefill}} \times \text{Prefill Time}) + (P_{\text{decode}} \times \text{Decode Time})$$

---

## Project Structure

| File | What It Does |
| :--- | :--- |
| `data/power_bounds.json` | Stores GPU power bounds and workload rules so numbers aren't hardcoded. |
| `data/workload_trace.json` | Stores the synthesized 1,000-request queue created by `generate_trace.py`. |
| `generate_trace.py` | Generates 1,000 random requests with Poisson arrival times and non-English token multipliers. |
| `profile_phases.py` | Reads the trace file, calculates total execution time and energy in Joules/kWh. |
| `check_gpu.py` | Checks Python and CUDA setup, confirming CPU simulation mode if no local GPU is found. |

---

## Key Papers Used

* **Splitwise (Patel et al., ISCA '24) & DistServe (Zhong et al., OSDI '24):** Grounding for splitting Prefill (compute-bound) vs Decode (memory-bound) power draws.
* **Sarathi-Serve (Agrawal et al., OSDI '24):** Reference for prompt chunking and request batching behavior.
* **ML.ENERGY Benchmark (2025):** Source for baseline empirical GPU energy telemetry metrics.
# Software-Driven LLM Energy Simulation Architecture (`llm-energy-sim`)

This repository evaluates energy-latency trade-offs in LLM inference serving by simulating chunked scheduling, paged memory management, and iteration-level dynamic frequency scaling.

---

## 1. Hardware Calibration & Workload Schema (`data/`)

* **`power_bounds.json`**: Grounded in empirical GPU telemetry (NVIDIA A100/H100 profile bounds).
  * `hardware_calibration.prefill_power_watts`: `250.0` (Compute-bound saturation at TDP).
  * `hardware_calibration.decode_power_watts`: `135.0` (Memory-bound frequency down-clock target).
  * `hardware_calibration.decode_power_baseline_watts`: `250.0` (Static unthrottled maximum baseline).
  * `hardware_calibration.dvfs_penalty_seconds`: `0.015` (15ms physical NVML driver delay per transition).
* **`workload_trace.json`**: Synthesized 1,000-request Poisson arrival queue containing `prompt_tokens` and `decode_tokens`.

---

## 2. Policy Engine Architecture

* **`baseline_policy.py` (`run_baseline_policy`)**:
  * Simulates standard serving framework pinned at static maximum TDP ($250\text{W}$) across prefill and decode phases.
  * Timing: $0.05\text{ ms/token}$ prefill ($20{,}000\text{ tok/s}$), $18.2\text{ ms/token}$ unthrottled decode baseline.
  * Zero frequency transition delay ($0.0\text{ ms}$).
* **`unified_policy.py` (`run_unified_policy`)**:
  * Simulates multi-layer co-optimization combining chunked prefill (Sarathi-Serve), paged memory allocation (vLLM), and iteration-level DVFS down-clocking during decode (throttLL'eM / Festina).
  * Timing: $0.05\text{ ms/token}$ prefill ($250\text{W}$), $18.8\text{ ms/token}$ memory-bound decode ($135\text{W}$, $<5\%$ SLO impact).
  * Injects a calibrated $15\text{ms}$ ($0.015\text{s}$) latency overhead per request to model NVML driver transition costs.

---

## 3. Experiment Orchestrator (`run_experiments.py`)

Executes comparative benchmarks across the workload trace and calculates key performance indicators:
* **Energy Consumption**: Joules ($\text{J}$), kilowatt-hours ($\text{kWh}$), and Joules per token ($\text{J/token}$).
* **Latency Performance**: Inter-Token Latency ($\text{ITL}$ in ms) and total simulation wall-clock time.
* **Exports**: Structured comparative JSON artifact saved to `data/simulation_results.json`.

---

## 4. Execution Pipeline

```bash
# 1. Generate workload trace
python generate_trace.py

# 2. Execute comparative evaluation
python run_experiments.py
```'''
---

## Key Papers Used

* **Sarathi-Serve (Agrawal et al., OSDI '24):** Chunked prefill scheduling and iteration-level batching dynamics.
* **vLLM (Kwon et al., SOSP '23):** PagedAttention memory management foundation.
* **throttLL’eM (Kakolyris et al., arXiv '25) & Festina (arXiv '26):** Iteration-level DVFS frequency scaling (250W→135W) and physical 15ms NVML driver latency penalty modeling.
* **Splitwise (Patel et al., ISCA '24) & DistServe (Zhong et al., OSDI '24):** Disaggregated prefill/decode phase execution characteristics.
* **ML.ENERGY Benchmark Suite (NeurIPS '25):** Baseline GPU power bounds and empirical energy telemetry.
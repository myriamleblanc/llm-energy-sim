# Project Build Log

**Project:** `llm-energy-sim` (LLM Inference Energy Simulator)  
**Author:** Myriam Leblanc  
**Supervisor:** Dr. Yasir Malik (NSERC Summer Project)

---

## Log Entries

### Entry 001: Project Setup (Aug 5, 2026)
* Set up the project folder `llm-energy-sim/` and initialized git.
* Added a `.gitignore` file to ignore `venv/`, cache folders, and generated JSON files so the repo stays clean.

---

### Entry 002: Virtual Environment (Aug 5, 2026)
* Created a Python virtual environment (`python3 -m venv venv`).
* Installed `numpy` and `simpy` to help with random request distributions and discrete-event simulation logic.

---

### Entry 003: Core Simulator Implementation (Aug 6, 2026)
* Created `data/power_bounds.json` to store static power bounds ($P_{\text{prefill}} = 250\text{W}$ and $P_{\text{decode}} = 135\text{W}$).
* Wrote `generate_trace.py` to synthesize 1,000 requests arriving in a Poisson queue, including a $1.30\times$ token multiplier for non-English prompts.
* Wrote `profile_phases.py` to calculate latency and energy consumption (in Joules) for each phase.
* Wrote `check_gpu.py` to verify system setup and fallback to CPU simulation mode.

### Entry 004: Modular Policy Refactoring & Comparative Experiment Orchestrator [2026-08-17]

#### Objective
Refactor the simulation architecture into modular policy components (`baseline_policy.py` and `unified_policy.py`) and implement a master evaluation runner (`run_experiments.py`) grounded directly in empirical literature and presentation milestones. Clean repository build artifacts to maintain strict standard-library isolation.

#### Added & Updated Files
* `data/power_bounds.json`: Injected `dvfs_penalty_seconds: 0.015` (15ms driver delay) and `decode_power_baseline_watts: 250.0` into the `hardware_calibration` schema.
* `baseline_policy.py`: Created modular baseline policy runner modeling standard static unthrottled serving ($250\text{W}$ prefill / $250\text{W}$ decode, $18.2\text{ms}$ baseline ITL, $0.0\text{ms}$ driver overhead). Grounded in Sarathi-Serve and vLLM literature.
* `unified_policy.py`: Created modular unified co-optimization policy runner modeling phase-aware iteration-level DVFS down-clocking ($250\text{W}$ prefill / $135\text{W}$ decode, $18.8\text{ms}$ throttled ITL, $15\text{ms}$ NVML driver penalty). Grounded in throttLL'eM, Festina, and ML.ENERGY benchmark telemetry.
* `run_experiments.py`: Implemented master evaluation orchestrator to ingest workload traces, execute both policies side-by-side, render the comparative verification matrix, and export `data/simulation_results.json`.
* `.gitignore`: Created ignore rules for `venv/`, cache/bytecode, and generated data artifacts (`workload_trace.json`, `simulation_results.json`). Cleaned untracked dependencies from remote Git history to restore 100% Python repository language statistics.
* `docs/DOCS.md`: Updated mathematical formulations, project structure table, and literature references to reflect the comparative policy framework.
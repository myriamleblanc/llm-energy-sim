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
"""
========================================================================================
BASELINE POLICY MODULE: Standard Serving Framework (Static Frequency / Max TDP)
========================================================================================

LITERATURE GROUNDING:
- Sarathi-Serve (Agrawal et al., OSDI '24): Chunked prefill execution at static max clock.
- vLLM (Kwon et al., SOSP '23): Standard paged KV-cache allocation.
- ML.ENERGY Benchmark (NeurIPS '25): Unthrottled empirical power & timing baseline bounds.
========================================================================================
"""

def run_baseline_policy(requests, p_prefill_watts, p_decode_baseline_watts):
    """
    Executes the 1,000-request workload trace under the Fixed Baseline Policy.
    """
    # Timing constants aligned with presentation slide benchmarks
    PREFILL_SEC_PER_TOKEN = 0.00005  # 0.05 ms/token (Compute-bound saturation at 250W)
    DECODE_SEC_PER_TOKEN_BASE = 0.0182  # 18.2 ms/token (Unthrottled decode at 250W)

    total_execution_time_sec = 0.0
    total_energy_joules = 0.0

    for req in requests:
        t_prefill = req["prompt_tokens"] * PREFILL_SEC_PER_TOKEN
        e_prefill = p_prefill_watts * t_prefill

        t_decode = req["decode_tokens"] * DECODE_SEC_PER_TOKEN_BASE
        e_decode = p_decode_baseline_watts * t_decode

        total_execution_time_sec += (t_prefill + t_decode)
        total_energy_joules += (e_prefill + e_decode)

    total_energy_kwh = total_energy_joules / 3600000.0
    mean_itl_ms = DECODE_SEC_PER_TOKEN_BASE * 1000.0

    return {
        "policy_name": "Fixed Baseline (Static Max TDP)",
        "total_time_sec": total_execution_time_sec,
        "total_energy_joules": total_energy_joules,
        "total_energy_kwh": total_energy_kwh,
        "mean_itl_ms": mean_itl_ms,
        "peak_decode_wattage": p_decode_baseline_watts,
        "dvfs_penalty_ms": 0.0
    }
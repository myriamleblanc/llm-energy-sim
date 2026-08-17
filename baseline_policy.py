# ROLE: Evaluates the workload trace under the standard Fixed-Power Baseline (Static 250W).

def run_baseline(requests, p_prefill, p_decode_baseline):
    PREFILL_SEC_PER_TOKEN = 0.00005  # 0.05 ms/token
    DECODE_SEC_PER_TOKEN = 0.0020    # 2.00 ms/token

    total_time_sec = 0.0
    total_energy_joules = 0.0

    for req in requests:
        t_prefill = req["prompt_tokens"] * PREFILL_SEC_PER_TOKEN
        t_decode = req["decode_tokens"] * DECODE_SEC_PER_TOKEN

        total_time_sec += (t_prefill + t_decode)
        total_energy_joules += (p_prefill * t_prefill) + (p_decode_baseline * t_decode)

    return {
        "time_sec": total_time_sec,
        "energy_joules": total_energy_joules,
        "energy_kwh": total_energy_joules / 3600000.0
    }
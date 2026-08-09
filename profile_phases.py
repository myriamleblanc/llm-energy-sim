#ROLE: This script reads the workload trace, calculates phase latencies, and computes energy in Joules and kWh.

import json  # Used to read power_bounds.json and workload_trace.json
import os    # Used to construct file paths safely across operating systems

def run_simulation():
    # 1. Load hardware power bounds from config file
    config_path = os.path.join("data", "power_bounds.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Store static power values in Watts (Prefill = 250W, Decode = 135W)
    p_prefill = config["hardware_calibration"]["prefill_power_watts"]
    p_decode = config["hardware_calibration"]["decode_power_watts"]

    # 2. Load the 1,000-request workload trace
    trace_path = os.path.join("data", "workload_trace.json")
    with open(trace_path, "r") as f:
        requests = json.load(f)

    # Processing speed constants (seconds per token)
    PREFILL_SEC_PER_TOKEN = 0.00005  # Fast matrix math speed (0.05 ms/token)
    DECODE_SEC_PER_TOKEN = 0.002     # Memory-bound generation speed (2.0 ms/token)

    # Accumulator variables for total metrics across all 1,000 requests
    total_energy_joules = 0.0
    total_time_seconds = 0.0

    # 3. Calculate latency and energy for every request
    for req in requests:
        # Calculate execution duration (Time = Tokens * Speed)
        prefill_time = req["prompt_tokens"] * PREFILL_SEC_PER_TOKEN
        decode_time = req["decode_tokens"] * DECODE_SEC_PER_TOKEN
        
        # Calculate energy consumed in Joules (Energy = Power * Time)
        prefill_energy = p_prefill * prefill_time
        decode_energy = p_decode * decode_time
        
        # Accumulate metrics into grand totals
        total_energy_joules += (prefill_energy + decode_energy)
        total_time_seconds += (prefill_time + decode_time)

    # 4. Print final simulation results summary
    print("=== SIMULATION RESULTS SUMMARY ===")
    print(f"Total Requests Processed : {len(requests)}")
    print(f"Total Execution Time     : {total_time_seconds:.2f} seconds")
    print(f"Total Energy Consumed    : {total_energy_joules:.2f} Joules")
    print(f"Energy in Kilowatt-Hours : {(total_energy_joules / 3600000.0):.6f} kWh")
    print("==================================")

# Guarantees execution only when run directly from the terminal
if __name__ == "__main__":
    run_simulation()
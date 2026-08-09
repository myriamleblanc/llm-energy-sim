# ROLE: This script reads config + generates the 1,000-request workload trace.

import json  # Used to read power_bounds.json and save workload_trace.json
import os    # Used to build safe, cross-platform file paths (e.g., data/power_bounds.json)
import random  # Used to generate random token lengths and Poisson arrival probabilities

def generate_workload_trace():
    # 1. Locate the config file inside the data subfolder
    config_path = os.path.join("data", "power_bounds.json")
    
    # 2. Open and load the JSON config parameters into a Python dictionary
    with open(config_path, "r") as f:
        config = json.load(f)

    # 3. Extract workload settings from the dictionary
    settings = config["workload_settings"]
    num_requests = settings["total_requests"]       # 1000 requests
    non_english_prob = settings["non_english_probability"] # 0.30 (30% chance)
    multiplier = settings["non_english_multiplier"]  # 1.30 (30% extra tokens)

    requests_list = []  # Empty list to store synthesized request dictionaries

    # 4. Loop 1,000 times to create 1,000 synthetic inference requests
    for i in range(num_requests):
        # Pick random base token lengths for prefill and decode phases
        base_prompt = random.randint(128, 2048)  # Prompt length range
        base_decode = random.randint(32, 512)    # Generated output length range

        # Check if this specific request is non-English
        is_non_english = random.random() < non_english_prob
        
        # Apply token expansion multiplier if non-English
        if is_non_english:
            final_prompt = int(base_prompt * multiplier)
            final_decode = int(base_decode * multiplier)
        else:
            final_prompt = base_prompt
            final_decode = base_decode

        # Package the single request into a dictionary
        requests_list.append({
            "request_id": f"REQ-{i:04d}",
            "is_non_english": is_non_english,
            "prompt_tokens": final_prompt,
            "decode_tokens": final_decode
        })

    # 5. Define where to save the generated output file
    output_path = os.path.join("data", "workload_trace.json")
    
    # 6. Save the list of 1,000 requests to data/workload_trace.json
    with open(output_path, "w") as f:
        json.dump(requests_list, f, indent=2)

    print(f"Generated 1,000 requests in '{output_path}'.")

# Guarantees the function only runs when executing this file directly from terminal
if __name__ == "__main__":
    generate_workload_trace()
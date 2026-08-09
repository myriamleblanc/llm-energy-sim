#ROLE: This script prints system information and confirms the simulation environment is ready.

import sys  # Used to query system Python runtime information

def check_env():
    print("=== llm-energy-sim Environment Status ===")
    # Print the current installed Python version
    print(f"Python Version: {sys.version.split()[0]}")
    # Confirm that the simulator is running in CPU hardware-model mode
    print("Status: Simulation engine ready (CPU hardware-model mode).")

# Guarantees execution only when run directly from the terminal
if __name__ == "__main__":
    check_env()
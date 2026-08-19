"""
Plotting script for Pareto Frontier: ITL Latency vs. Total Energy (kWh).
Reads evaluated points directly from data/simulation_results.json.
"""

import json
import os
import matplotlib.pyplot as plt

def generate_pareto_plot():
    results_path = os.path.join("data", "simulation_results.json")
    if not os.path.exists(results_path):
        print(f"Error: Could not find '{results_path}'. Run run_experiments.py first.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    sweep = data.get("pareto_frontier_sweep", [])
    if not sweep:
        print("Error: No 'pareto_frontier_sweep' key found in JSON.")
        return

    # Extract coordinates
    itl_values = [pt["mean_itl_ms"] for pt in sweep]
    energy_values = [pt["total_energy_kwh"] for pt in sweep]
    labels = [pt["label"] for pt in sweep]
    savings = [pt["energy_savings_pct"] for pt in sweep]

    # Plot setup
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(itl_values, energy_values, color="#2563EB", linestyle="--", linewidth=2, zorder=2, label="Pareto Curve")
    plt.scatter(itl_values, energy_values, color="#1D4ED8", s=70, zorder=3, edgecolors="black", linewidth=0.8)

    # Annotate points
    for i, label in enumerate(labels):
        if "135W" in label:
            # Highlight target policy
            plt.scatter(itl_values[i], energy_values[i], color="#10B981", s=130, zorder=4, edgecolors="black")
            plt.annotate(
                f"Optimal Point ({label})\n-{savings[i]:.1f}% Energy",
                (itl_values[i], energy_values[i]),
                textcoords="offset points",
                xytext=(25, -5),
                fontweight="bold",
                color="#065F46",
                arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2)
            )
        else:
            plt.annotate(
                f"{label}\n({energy_values[i]:.4f} kWh)",
                (itl_values[i], energy_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8
            )

    # 5% Latency SLO Threshold Reference Line
    slo_limit = sweep[0]["mean_itl_ms"] * 1.05
    plt.axvline(x=slo_limit, color="#DC2626", linestyle=":", linewidth=1.5, label="5% ITL SLO Budget")

    # Titles and formatting
    plt.title("Multi-Objective Pareto Frontier: User Latency vs. Grid Energy", fontsize=11, fontweight="bold", pad=18)
    plt.ylim(0.16, 0.40) 
    plt.xlabel("Mean Inter-Token Latency (ITL ms) — Lower is Better", fontsize=9)
    plt.ylabel("Total Energy Consumption (kWh) — Lower is Better", fontsize=9)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save to docs folder
    output_path = os.path.join("docs", "pareto_frontier.png")
    os.makedirs("docs", exist_ok=True)
    plt.savefig(output_path)
    print(f"Chart saved successfully to '{output_path}'.")

if __name__ == "__main__":
    generate_pareto_plot()
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Data from Phase 3 Benchmarks ---
threads = [1, 2, 4, 8, 16]
x_indexes = np.arange(len(threads))
bar_width = 0.35

# Workload C: 100% Read, Uniform Distribution
btree_c = [1.11, 1.85, 3.15, 2.69, 2.74]
hydrods_c = [1.87, 3.42, 6.25, 15.04, 21.41]

# Workload B: 95% Read / 5% Insert, Zipfian Skew
btree_b = [2.15, 1.07, 0.81, 0.87, 0.81]
hydrods_b = [4.07, 6.95, 6.10, 8.37, 10.45]

# --- Setup Plot Styling ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot Workload C ---
ax1.bar(x_indexes - bar_width/2, hydrods_c, bar_width, label='HydroDS (Wait-Free)', color='#2ca02c')
ax1.bar(x_indexes + bar_width/2, btree_c, bar_width, label='B-Tree (Global Lock)', color='#d62728')

ax1.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax1.set_ylabel('Throughput (MOps/s)', fontsize=12, fontweight='bold')
ax1.set_title('Workload C: 100% Read (Uniform Distribution)', fontsize=14, pad=15)
ax1.set_xticks(x_indexes)
ax1.set_xticklabels(threads)
ax1.legend(loc='upper left', frameon=True, shadow=True)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# --- Plot Workload B ---
ax2.bar(x_indexes - bar_width/2, hydrods_b, bar_width, label='HydroDS (Pressure-Flow)', color='#1f77b4')
ax2.bar(x_indexes + bar_width/2, btree_b, bar_width, label='B-Tree (Global Lock)', color='#ff7f0e')

ax2.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax2.set_ylabel('Throughput (MOps/s)', fontsize=12, fontweight='bold')
ax2.set_title('Workload B: 95% Read / 5% Insert (Zipfian Skew)', fontsize=14, pad=15)
ax2.set_xticks(x_indexes)
ax2.set_xticklabels(threads)
ax2.legend(loc='upper left', frameon=True, shadow=True)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# --- Save and Show ---
output_file = "artifacts/scaling_graphs.png"
os.makedirs("artifacts", exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Graphs successfully saved to: {output_file}")

try:
    plt.show()
except Exception as e:
    print("Could not display interactive window (likely no display server found).")
    print("Please view the saved PNG file instead.")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Create artifacts directory if not exists
os.makedirs('artifacts', exist_ok=True)

# 1. Latency Plot
data_lat = {
    'Operation': ['Insert', 'Search', 'Range Small', 'Range Medium', 'Range Large', 'Delete'],
    'HydroDS': [7.5090, 0.4975, 0.0881, 0.1986, 0.0966, 0.7702],
    'CSB+-Tree': [7.0767, 0.7908, 0.1797, 0.4387, 0.2098, 1.2507],
    'ALEX': [3.6405, 0.0770, 0.0633, 0.2904, 0.1327, 0.2961]
}
df_lat = pd.DataFrame(data_lat)
df_lat.set_index('Operation').plot(kind='bar', figsize=(10, 6))
plt.title('Single-Threaded Core Operations Latency')
plt.ylabel('Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('artifacts/latency_plot.png')
plt.close()

# 2. Hardware Counters Plot
data_hw = {
    'Metric': ['L1 Misses(M)', 'Cache Misses(M)', 'Cache Refs(M)', 'Instrs(B)', 'Cycles(B)', 'Branches(B)', 'Branch Misses(M)', 'IPC'],
    'HydroDS': [141.7, 167.3, 206.9, 5.76, 7.08, 1.15, 77.3, 0.81],
    'CSB+-Tree': [72.8, 112.5, 134.9, 5.85, 6.85, 1.11, 85.3, 0.85],
    'ALEX': [30.0, 60.0, 75.8, 5.88, 3.20, 0.99, 22.6, 1.84]
}
df_hw = pd.DataFrame(data_hw)
df_hw.set_index('Metric').plot(kind='bar', figsize=(12, 6))
plt.title('Cache-Miss & Hardware Counters Analysis (1 Thread)')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('artifacts/hardware_counters_plot.png')
plt.close()

# 3. Memory Footprint Plot
data_mem = {
    'Structure': ['HydroDS', 'CSB+-Tree', 'ALEX'],
    'Memory (MB)': [33, 33, 89]
}
plt.figure(figsize=(6, 4))
plt.bar(data_mem['Structure'], data_mem['Memory (MB)'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Memory Footprint Comparison (1 Thread)')
plt.ylabel('Memory (MB)')
plt.tight_layout()
plt.savefig('artifacts/memory_footprint_plot.png')
plt.close()

# 4. Scaling Plots
threads = [1, 2, 4, 8, 16, 32]
hydrods_ins = [7.5090, 7.2150, 7.8671, 8.3546, 8.5909, 8.8903]
hydrods_src = [0.4975, 0.3369, 0.2561, 0.2533, 0.2677, 0.2472]
csb_ins = [7.0767, 12.6073, 15.8659, 16.2451, 23.6096, 22.3218]
csb_src = [0.7908, 0.4349, 0.2527, 0.2680, 0.2561, 0.2564]
alex_ins = [3.6405, 7.4280, 9.1711, 10.4683, 24.0261, 24.1301]
alex_src = [0.0770, 0.1964, 0.2355, 0.2669, 0.2242, 0.2180]

plt.figure(figsize=(8, 5))
plt.plot(threads, hydrods_ins, marker='o', label='HydroDS')
plt.plot(threads, csb_ins, marker='s', label='CSB+-Tree')
plt.plot(threads, alex_ins, marker='^', label='ALEX')
plt.title('Thread Scaling: Insert Latency')
plt.xlabel('Threads')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('artifacts/scaling_insert_plot.png')
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(threads, hydrods_src, marker='o', label='HydroDS')
plt.plot(threads, csb_src, marker='s', label='CSB+-Tree')
plt.plot(threads, alex_src, marker='^', label='ALEX')
plt.title('Thread Scaling: Search Latency')
plt.xlabel('Threads')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('artifacts/scaling_search_plot.png')
plt.close()

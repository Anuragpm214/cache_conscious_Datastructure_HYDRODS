# HydroDS: A Cache-Conscious Concurrent Data Structure

HydroDS is a highly optimized, concurrent, and cache-conscious in-memory index structure designed to bridge the gap between traditional B-Trees and modern Learned Indexes. By leveraging contiguous memory blocks (buckets) and AVX2 SIMD vectorization, HydroDS provides deterministic performance, low memory footprint, and massive scalability.

## 🚀 Extreme Performance Profiling (ALEX vs CSB+-Tree vs HydroDS)

The following benchmarks were recorded on bare-metal hardware using Linux `perf` counters (`taskset -c 0 perf stat -e cache-misses,L1-dcache-load-misses`).

**Workload Profile:**
- **Keys:** 5 Million (32-bit Integers)
- **Queries:** 500k Searches, 50k Small Ranges (Length 100), 500k Deletes
- **Configuration:** HydroDS (Capacity = 256), CSB+-Tree (tlx), ALEX (Learned Index)

### 📊 1. Core Operations Latency (Seconds)

| Operation | HydroDS (C=256) | CSB+-Tree (tlx) | ALEX (Learned Index) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Insert (5M)** | 4.38 s | 5.07 s | **2.33 s** | ALEX dominates writes on uniform data, but HydroDS beats CSB+. |
| **Search (500k)** | 0.186 s | 0.543 s | **0.039 s** | HydroDS is **3x faster** than CSB+-Tree. ALEX's $O(1)$ ML prediction wins. |
| **Range Scan (Small)**| 0.041 s | 0.185 s | **0.023 s** | HydroDS is **4.5x faster** than CSB+-Tree due to contiguous layout. |
| **Delete (500k)** | 0.619 s | 0.747 s | **0.099 s** | HydroDS outperforms CSB+ due to its vector-free `Pressure-Flow` merging. |

*(Note: Medium/Large range queries for CSB+ and ALEX were omitted from this table due to GCC `-O3` dead-code elimination, but HydroDS strictly evaluated them in 0.043s and 0.023s respectively).*

---

### 🧠 2. The Cache-Miss Paradox (Why HydroDS beats CSB+)

When analyzing hardware counters, a fascinating architectural truth emerges:

| Structure | L1 D-Cache Misses | Time Taken (Search) |
| :--- | :--- | :--- |
| **CSB+-Tree** | 67.6 Million | 0.543 s (Slow) |
| **HydroDS** | 130.5 Million | **0.186 s (Fast)** |

**The Paradox:** How is HydroDS 3x faster than CSB+-Tree when it has 2x more L1 Cache Misses?
**The Answer: Memory Access Patterns & Prefetching.** 
CSB+-Tree performs *pointer-chasing*. Every node traversal is a dependent memory fetch, stalling the CPU pipeline until the RAM responds. 
HydroDS performs *linear SIMD scans* over 1KB contiguous buckets. While scanning 1KB triggers multiple L1 load misses, the CPU's Hardware Prefetcher sees the linear pattern and fetches the data *before* the CPU needs it. The misses are fully pipelined, resulting in vastly superior wall-clock speeds.

---

### 💾 3. The Memory Footprint Tax

Learned Indexes (like ALEX) achieve blistering point-query speeds by predicting array locations. However, this requires maintaining massive sparse arrays with empty "gaps". 

| Structure | Physical RAM Used | Bytes per Key |
| :--- | :--- | :--- |
| **ALEX** | 88 MB | 18.45 bytes/key |
| **CSB+-Tree** | 33 MB | 6.92 bytes/key |
| **HydroDS** | **32 MB** | **6.71 bytes/key** |

**Conclusion:** HydroDS provides a deterministic memory footprint that is **2.75x smaller than ALEX** and even edges out the CSB+-Tree! HydroDS proves that you don't need to sacrifice massive amounts of RAM (like Learned Indexes do) to beat B-Trees.

---

## 🛠️ Compilation and Benchmarking

This repository contains a unified benchmarking suite designed for strict hardware profiling.

### 1. Build the Benchmark
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 2. Run with Performance Counters
To replicate the hardware-level cache analysis, use `taskset` to pin execution to a single core and `perf` to track L1 misses:

```bash
# Profile HydroDS
taskset -c 0 perf stat -e cache-misses,L1-dcache-load-misses ./build/master_benchmark hydrods

# Profile CSB+-Tree
taskset -c 0 perf stat -e cache-misses,L1-dcache-load-misses ./build/master_benchmark csb_tree

# Profile ALEX
taskset -c 0 perf stat -e cache-misses,L1-dcache-load-misses ./build/master_benchmark alex
```

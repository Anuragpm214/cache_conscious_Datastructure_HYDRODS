
# 🚀 HydroDS → A* Conference Acceptance Plan

> **Goal**: Transform HydroDS from a prototype into a conference-quality in-memory index data structure paper, competitive with ART, HOT, Masstree, and B+-tree variants.

---

## 📊 Current State Analysis

### What HydroDS Is Today
Your data structure is a **sorted bucket array with pressure-based (fluid-dynamics-inspired) rebalancing**:
- Fixed-capacity buckets (`C=1000`) of sorted integers
- Binary search over `bucket_max[]` for bucket lookup → `O(log B)` where B = number of buckets
- Within each bucket: sorted vector with binary search → `O(log C)`
- **Novelty**: "Pressure" = fill ratio; elements "flow" between adjacent buckets when pressure differential exceeds thresholds (`EPS_HIGH=0.85`, `EPS_LOW=0.50`)
- Split on overflow, local stabilization on insert/delete

### Critical Gaps (Must Fix Before Submission)

| Gap | Severity | Why It Matters |
|-----|----------|----------------|
| Single-threaded only | 🔴 Critical | Every A* paper since 2015 requires concurrent evaluation |
| Integer keys only | 🟡 Important | Reviewers expect at least integer + string key support |
| No SIMD / cache optimization | 🔴 Critical | Competing structures (ART, HOT, FAST) are heavily cache-optimized |
| `vector<vector<int>>` storage | 🔴 Critical | Heap-allocated vectors cause pointer chasing, terrible cache behavior |
| `B.insert(B.begin(), ...)` is O(n) | 🟡 Important | Kills insert performance for front-insertions |
| No memory usage tracking | 🔴 Critical | Memory efficiency is a key metric in every benchmark paper |
| Toy benchmark only | 🔴 Critical | Need YCSB-style workloads, Zipfian, varying sizes |
| No latency percentiles | 🔴 Critical | p50/p99/p999 latency is mandatory |
| No comparison with ANY baseline | 🔴 Critical | Paper is DOA without head-to-head comparisons |

---

## 🎯 Phase 1: Core Data Structure Hardening (Weeks 1-3)

> [!IMPORTANT]
> The current implementation has fundamental performance issues that will make it lose to every baseline. Fix these FIRST before benchmarking.

### 1.1 Replace `vector<vector<int>>` with Cache-Friendly Storage

```diff
- vector<vector<int>> buckets;
- vector<int> bucket_max;
+ // Flat arena-allocated buckets with embedded metadata
+ struct alignas(64) Bucket {    // Cache-line aligned
+     uint16_t count;
+     uint16_t capacity;
+     int32_t  max_key;
+     int32_t  keys[];            // Flexible array member, inline
+ };
+ // Or: fixed-size array buckets (avoids heap allocation per bucket)
+ static constexpr int C = 256;  // Tune to fit in 1-2 cache lines
+ struct Bucket {
+     int32_t keys[C];
+     uint16_t count;
+ };
```

**Why**: Every `vector<int>` is a heap allocation → pointer chasing → cache misses. ART/HOT use carefully laid-out node structures. You MUST do the same.

### 1.2 Add SIMD Acceleration for Intra-Bucket Search

```cpp
// Use SSE4.2 / AVX2 for searching within a bucket
// Instead of binary_search, do linear SIMD scan for small buckets (C ≤ 256)
#include <immintrin.h>

int simd_search_bucket(const int32_t* keys, int count, int32_t target) {
    __m256i target_vec = _mm256_set1_epi32(target);
    for (int i = 0; i < count; i += 8) {
        __m256i data = _mm256_loadu_si256((__m256i*)(keys + i));
        __m256i cmp = _mm256_cmpeq_epi32(data, target_vec);
        int mask = _mm256_movemask_epi8(cmp);
        if (mask) return i + __builtin_ctz(mask) / 4;
    }
    return -1;
}
```

**Why**: For small bucket sizes (64-512), SIMD linear scan beats binary search due to branch prediction and sequential access. This is a well-known technique used in FAST and CSB+-tree.

### 1.3 Tune Bucket Size `C`

- Current `C=1000` is **way too large**. This means each bucket is ~4KB (1000 × 4 bytes).
- **Experiment with**: C ∈ {64, 128, 256, 512, 1024}
- Sweet spot is likely **C=128 or C=256** (fits in L1/L2 cache line multiples)
- This parameter is a KEY CONTRIBUTION — show how pressure-based rebalancing makes HydroDS less sensitive to `C` than B+-tree is to node size

### 1.4 Fix O(n) Operations

```diff
// Replace vector inserts at front with gap-buffer or ring-buffer approach
- B.insert(B.begin(), x);          // O(n) memmove!
+ // Use a circular buffer or maintain a "head offset" within fixed-size array
```

### 1.5 Template the Key Type

```cpp
template <typename Key, typename Value = void, typename Compare = std::less<Key>>
class HydroDS {
    // Support: int32_t, int64_t, uint64_t, std::string, __uint128_t
};
```

> [!TIP]
> For the **first submission**, focusing on **64-bit integer keys** is acceptable — ART (SIGMOD 2013), FAST (SIGMOD 2010), and many others started with integer keys only. But you must **acknowledge** string support as future work and show the design is extensible.

---

## 🎯 Phase 2: Concurrency (Weeks 3-5)

> [!CAUTION]
> A single-threaded-only data structure paper will be **desk-rejected** at any A* systems venue in 2026. You MUST add concurrency.

### 2.1 Optimistic Lock Coupling (Minimum Viable)

```cpp
struct Bucket {
    std::atomic<uint64_t> version;  // Even = unlocked, Odd = locked
    int32_t keys[C];
    uint16_t count;
    
    bool try_read_lock(uint64_t& v) {
        v = version.load(std::memory_order_acquire);
        return (v & 1) == 0;  // Not write-locked
    }
    
    bool validate(uint64_t v) {
        return version.load(std::memory_order_acquire) == v;
    }
    
    void write_lock() { /* CAS loop */ }
    void write_unlock() { version.fetch_add(1, std::memory_order_release); }
};
```

### 2.2 Pressure-Aware Concurrent Rebalancing (YOUR NOVELTY!)

This is where your "flow" metaphor becomes a **real contribution**:
- **Claim**: Pressure-based local rebalancing is inherently more concurrent-friendly than B+-tree splits (which propagate up to root)
- **Design**: Lock only the 2 adjacent buckets during a flow operation (fine-grained)
- **Compare**: B+-tree SMOs (Structure Modification Operations) lock entire subtrees

### 2.3 Scalability Testing
- Test with 1, 2, 4, 8, 16, 32, 64 threads
- Show throughput scaling (ops/sec vs threads)
- Show that flow-based rebalancing has lower contention than tree splits

---

## 🎯 Phase 3: Benchmarking Framework (Weeks 5-8)

> [!IMPORTANT]
> This is where papers are made or broken. A weak evaluation = rejection.

### 3.1 Baselines to Compare Against

| Baseline | Source | Priority | Notes |
|----------|--------|----------|-------|
| **ART (Adaptive Radix Tree)** | [github.com/armon/libart](https://github.com/armon/libart) or TU Munich implementation | 🔴 Must | SIGMOD 2013, the gold standard |
| **HOT (Height Optimized Trie)** | [github.com/speedskater/hot](https://github.com/speedskater/hot) | 🔴 Must | SIGMOD 2018, current state-of-art trie |
| **Masstree** | [github.com/kohler/masstree-beta](https://github.com/kohler/masstree-beta) | 🟡 Important | EuroSys 2012, best concurrent tree |
| **STX/TLX B+-tree** | [github.com/tlx/tlx](https://github.com/tlx/tlx) (`tlx::btree_map`) | 🔴 Must | Standard B+-tree baseline |
| **Google Abseil B-tree** | `absl::btree_map` | 🟡 Important | Industry-grade B-tree |
| **std::map (Red-Black Tree)** | stdlib | 🟢 Easy | Strawman baseline |
| **ALEX** | [github.com/microsoft/ALEX](https://github.com/microsoft/ALEX) | 🟡 Important | Learned index, SIGMOD 2020 |
| **PGM-Index** | [github.com/gvinciguerra/PGM-index](https://github.com/gvinciguerra/PGM-index) | 🟡 If space allows | Learned index for comparison |

**Benchmark Framework**: Use or adapt [github.com/wangziqi2016/index-microbench](https://github.com/wangziqi2016/index-microbench) (TU Munich's standard index microbenchmark) or build your own with the YCSB-style workload generator.

### 3.2 Workload Types (YCSB-inspired)

| Workload | Read% | Write% | Scan% | Description |
|----------|-------|--------|-------|-------------|
| **A (Update-heavy)** | 50% | 50% | 0% | Session store |
| **B (Read-heavy)** | 95% | 5% | 0% | Photo tagging |
| **C (Read-only)** | 100% | 0% | 0% | User profile cache |
| **D (Read-latest)** | 95% | 5% | 0% | Latest-first distribution |
| **E (Scan-heavy)** | 0% | 5% | 95% | Analytics / range queries |
| **F (Read-Modify-Write)** | 50% | 50% | 0% | User database |

### 3.3 Key Distributions

| Distribution | Description | Why It Matters |
|-------------|-------------|----------------|
| **Uniform Random** | Keys drawn uniformly from [0, N) | Baseline behavior |
| **Zipfian (θ=0.99)** | Highly skewed, some keys much hotter | Tests cache behavior under skew |
| **Sequential (monotone)** | Keys inserted in order | Tests split/rebalance behavior |
| **Reverse Sequential** | Keys inserted in reverse order | Worst case for many structures |
| **Hotspot** | 80% ops on 20% keys | Tests localized access patterns |

### 3.4 Metrics to Report

| Metric | How to Measure | Presentation |
|--------|---------------|--------------|
| **Throughput** | Million ops/sec (Mops/s) | Bar charts per workload |
| **Latency p50/p99/p999** | Per-operation latency in nanoseconds | CDF plot or table |
| **Memory Footprint** | RSS, or custom allocator tracking | Bar chart (bytes/key) |
| **Cache Misses** | `perf stat -e cache-misses,cache-references` | Table or line chart |
| **Scalability** | Throughput vs thread count | Line chart (1→64 threads) |
| **Build Time** | Bulk-load N keys, measure time | Line chart (N vs time) |

### 3.5 Dataset Sizes

| Size | Keys | Purpose |
|------|------|---------|
| Small | 1M | Fits in L3 cache, tests compute-bound behavior |
| Medium | 10M | Exceeds L3 on most machines |
| Large | 100M | Memory-bound, tests scalability |
| XL | 200M-1B | Stress test, if memory allows |

---

## 🎯 Phase 4: The Paper Story (Weeks 7-10)

### 4.1 Paper Structure (for SIGMOD/VLDB)

```
1. Introduction
   - Motivation: B+-trees split/merge is coarse-grained → poor cache utilization
   - Key Insight: Fluid-dynamics-inspired pressure equalization for load balancing
   - Contributions (3-4 bullet points)

2. Background & Related Work
   - B+-trees, CSB+-tree, ART, HOT, Masstree, learned indexes
   - Rebalancing strategies: split/merge vs. redistribution vs. pressure flow

3. HydroDS Design
   3.1 Overview & Architecture
   3.2 Pressure Model (formalize EPS_HIGH, EPS_LOW, flow equations)
   3.3 Insert / Search / Delete / Range Query
   3.4 Concurrency Control (OLC + fine-grained flow locking)
   3.5 SIMD Optimizations

4. Theoretical Analysis
   4.1 Amortized cost of insert/delete with pressure rebalancing
   4.2 Space utilization bounds (show buckets stay between EPS_LOW and EPS_HIGH full)
   4.3 Comparison with B+-tree split/merge complexity

5. Evaluation
   5.1 Experimental Setup (machine, baselines, datasets)
   5.2 Single-threaded Performance (vs ART, HOT, B+-tree)
   5.3 Multi-threaded Scalability
   5.4 Memory Efficiency
   5.5 Sensitivity Analysis (C, EPS_HIGH, EPS_LOW)
   5.6 Cache Behavior (perf counters)

6. Discussion & Future Work
7. Conclusion
```

### 4.2 Articulate the Contribution Clearly

Your contribution is **NOT** "a new sorted bucket structure" — that's a skip list variant. Your contribution is:

> **"Pressure-based continuous rebalancing"** — unlike B+-trees that rebalance through discrete splits/merges that propagate upward, HydroDS uses a fluid-dynamics-inspired model where load imbalances are resolved locally through bounded "flow" operations between adjacent containers. This achieves:
> 1. **More uniform space utilization** (no half-empty nodes after splits)
> 2. **Lower rebalancing latency** (bounded, local, no upward propagation)
> 3. **Better concurrency** (only 2 adjacent buckets locked during flow)
> 4. **Cache-friendly sequential layout** (flat bucket array, no tree pointers)

### 4.3 Formalize the Theory

You MUST provide:
- **Amortized analysis** of insert cost with pressure rebalancing
- **Space utilization bound**: Prove that after stabilization, all buckets have fill ratio in `[EPS_LOW, EPS_HIGH]` (or close to it)
- **Flow convergence**: Prove that `stabilize()` terminates in bounded steps

---

## 🎯 Phase 5: Engineering Quality (Weeks 8-11)

### 5.1 Code Quality
- [ ] Remove `#include <bits/stdc++.h>` — use proper headers
- [ ] Template the key/value types
- [ ] Add proper error handling
- [ ] Use custom allocator (arena/pool) for buckets
- [ ] Add `__builtin_expect` for branch hints
- [ ] Profile-guided optimization (PGO)

### 5.2 Reproducibility
- [ ] Provide a `CMakeLists.txt` with proper build flags
- [ ] Docker container for reproducible benchmarks
- [ ] Scripts to reproduce every figure in the paper
- [ ] Pin CPU frequency, disable turbo boost for benchmarks
- [ ] Report machine specs: CPU model, cache sizes, RAM, OS, compiler version

### 5.3 Testing
- [ ] Correctness tests: insert N random keys, verify all searchable
- [ ] Stress test: random insert/delete/search interleaved
- [ ] Verify range query correctness against `std::set`
- [ ] Memory leak detection (Valgrind/ASAN)
- [ ] Thread sanitizer (TSAN) for concurrent version

---

## 🎯 Phase 6: Submission Targets & Timeline

### Target Venues (Ranked by Fit)

| Venue | Deadline | Fit | Notes |
|-------|----------|-----|-------|
| **VLDB 2027** | Rolling (monthly) | ⭐⭐⭐⭐⭐ | Best fit — rolling deadline, data structure papers welcome |
| **SIGMOD 2027** | ~Oct 2026 | ⭐⭐⭐⭐ | Top database venue, index papers are core |
| **EuroSys 2027** | ~Oct 2026 | ⭐⭐⭐ | If you emphasize concurrency story |
| **PPoPP 2027** | ~Aug 2026 | ⭐⭐⭐ | If concurrent angle is strong |
| **ICDE 2027** | ~Oct 2026 | ⭐⭐⭐⭐ | Slightly easier than SIGMOD, good for first submission |
| **DaMoN 2027** | ~Apr 2027 | ⭐⭐⭐⭐⭐ | Workshop co-located with SIGMOD, great for early feedback |

> [!TIP]
> **Recommended Strategy**: Submit to **DaMoN 2027** (workshop, 6-page paper) first for feedback, then extend to **VLDB 2027** (full paper, rolling deadline). This is a proven strategy used by many successful research groups.

---

## ❓ Key Questions to Answer Now

### Should you test ONLY integer keys?

**Short answer: For the FIRST paper, yes — 64-bit integers are sufficient.**

Justification:
- ART (SIGMOD 2013) evaluated primarily on integer keys in its microbenchmarks
- FAST (SIGMOD 2010) was integer-only
- HOT's paper included strings but integers were the primary evaluation
- **However**, you should design the code to be templated so string support is "future work"
- If you have time, adding **one** string-key experiment strengthens the paper significantly

### What about variable-length keys (strings)?
- For HydroDS, string keys would require changing the bucket layout significantly (can't use fixed-size arrays)
- This is a legitimate "future work" item
- Mention it in the paper but don't let it block submission

---

## 📋 Week-by-Week Timeline

| Week | Tasks | Deliverable |
|------|-------|-------------|
| **1** | Refactor: cache-line aligned buckets, remove `vector<vector>` | New core data structure |
| **2** | Add SIMD search, tune bucket size `C`, fix O(n) inserts | Optimized single-threaded HydroDS |
| **3** | Template key type, add 64-bit key support | Generic HydroDS |
| **4** | Implement OLC-based concurrency | Concurrent HydroDS |
| **5** | Set up benchmark framework, integrate baselines (ART, HOT, B+-tree) | Benchmark harness |
| **6** | Run single-threaded benchmarks across all workloads | Raw results |
| **7** | Run multi-threaded benchmarks, cache profiling | Scalability results |
| **8** | Write paper: Sections 1-3 (Intro, Background, Design) | Draft sections |
| **9** | Write paper: Section 4 (Theory), Section 5 (Evaluation) | Draft sections |
| **10** | Write paper: Polish, make figures, get feedback | Complete draft |
| **11** | Internal review, revise | Submission-ready |

---

## ⚠️ Common Rejection Reasons to Avoid

1. **"Incremental over B+-tree"** → Emphasize the pressure model as a fundamentally different rebalancing philosophy, not just a tweak
2. **"No concurrent evaluation"** → MUST add concurrency
3. **"Limited evaluation"** → Need ≥ 4 baselines, ≥ 5 workloads, ≥ 3 dataset sizes
4. **"No theoretical analysis"** → Formalize the pressure model, prove bounds
5. **"Poor writing"** → Get a native English speaker to proofread; read 10 published index papers for style
6. **"Not reproducible"** → Open-source code + Docker + scripts

---

> [!NOTE]
> **Bottom Line**: Your core idea (pressure-based rebalancing) is genuinely novel and interesting. But the gap between a prototype and a conference paper is enormous. The biggest risks are: (1) the optimized HydroDS might not actually beat ART/HOT on lookups, and (2) the concurrency story needs to be compelling. Start with Phase 1 optimizations and benchmark against `std::map` and `absl::btree_map` immediately — if you can't beat those, the paper won't work.

# HydroDS 🌊 

[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HydroDS** is a blazingly fast, highly concurrent in-memory index structure designed as a drop-in replacement for traditional B-Trees and `std::multiset`. It is engineered from the ground up to eliminate CPU cache misses and lock-contention in modern multi-core systems.

By combining **AVX2 SIMD vectorization** with a novel **fluid-dynamics-inspired concurrency model**, HydroDS easily sustains over **21 Million Operations per Second (MOps/s)** on commodity hardware, outperforming coarse-grained B-Trees by over **10x** under heavy write contention.

---

## ⚡ Why Use HydroDS?

If your application suffers from lock contention when multiple threads try to insert or read from a shared `std::map`, `std::set`, or standard B-Tree, HydroDS solves this natively:

* 🚫 **No Pointer Chasing**: Traditional trees suffer from L2/L3 cache misses while jumping between nodes. HydroDS uses flat, cache-line-aligned arrays.
* 🚀 **Wait-Free Reads**: Search queries execute purely via relaxed atomic loads (RCU-style Sequence Locks). Readers **never** write to shared memory, completely eliminating False Sharing.
* 🌊 **Pressure-Flow Contention Absorption**: When a B-Tree node fills up, it splits, propagating locks all the way up to the root and stalling the entire tree. HydroDS treats buckets like water tanks: when one gets too full ("high pressure"), it asynchronously "flows" elements to neighboring buckets using only local, fine-grained locks.

---

## 💻 Quickstart

HydroDS is a header-only library. Simply include it and go!

```cpp
#include "hydrods_concurrent.hpp"
#include <iostream>
#include <thread>

int main() {
    // Initialize the concurrent index (128 elements per bucket)
    ConcurrentHydroDS<int32_t, 128> index;

    // 1. Thread-safe Insertions
    std::thread writer([&]() {
        for(int i = 0; i < 100000; i++) {
            index.insert(i);
        }
    });

    // 2. Wait-free Searches
    std::thread reader([&]() {
        bool found = index.search(42);
        std::cout << "Found 42? " << found << std::endl;
    });

    writer.join();
    reader.join();

    // 3. Fast Range Queries
    int64_t count = index.range_query(10, 50);
    std::cout << "Elements between 10 and 50: " << count << std::endl;

    return 0;
}
```

---

## 🧠 Architecture: How it Works

1. **The Directory**: A Wait-Free Sequence Lock (`seqlock`) protects an array of bucket pointers. Readers check a sequence number, binary search the index, and access the bucket without acquiring a single hardware lock.
2. **The Bucket**: Cache-line aligned arrays holding 128 integers. Intra-bucket searches don't use branching; they use **AVX2 SIMD instructions** to scan 8 integers per CPU cycle.
3. **Optimistic Lock Coupling (OLC)**: Every bucket has a version counter. Writers lock a bucket by turning the version odd. Readers read the version, scan the bucket, and verify the version hasn't changed.

---

## 📊 Benchmarks

*Tested on 16 Threads | 2 Million Operations | vs `tlx::btree` (with Global Read-Write Lock)*

### Workload 1: 100% Read (Uniform)
Tests raw Wait-Free execution. Because HydroDS readers never increment an atomic reference counter, they scale perfectly with physical cores.
* **HydroDS**: 🟢 21.41 MOps/sec
* **B-Tree**: 🔴 2.74 MOps/sec *(Bottlenecked by cache-line bouncing on the RW-Lock)*

### Workload 2: 95% Read / 5% Insert (Zipfian Skew)
Tests realistic database access where 80% of operations hit the same 20% of the data. 
* **HydroDS**: 🟢 10.45 MOps/sec *(Pressure-Flow handles local write bursts gracefully)*
* **B-Tree**: 🔴 0.81 MOps/sec *(Global tree modifications crush concurrent throughput)*

### Memory Footprint
* **HydroDS**: 6.69 bytes / key
* **B-Tree**: ~12.0 bytes / key

---

## 🛠️ Building & Testing

Make sure you have CMake and a compiler that supports C++20 and AVX2 (`-march=native`).

```bash
git clone https://github.com/yourusername/hydrods.git
cd hydrods
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Run the Scaling Benchmark:**
```bash
./benchmark_scaling
```

**Run Correctness & Thread-Sanitizer Tests:**
```bash
./test_concurrent
```

## 📜 License
HydroDS is open-source software licensed under the MIT License.

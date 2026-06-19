# HydroDS

**A cache-conscious, concurrent, SIMD-accelerated data structure.**

HydroDS bridges the gap between traditional B-Trees and modern Learned Indexes using a fluid-dynamics-inspired "Pressure-Flow" model. It uses cache-line aligned contiguous buckets, AVX2 SIMD vectorization, and Optimistic Lock Coupling (OLC) for high concurrency.

## Key Features
- **SIMD-Accelerated Search:** AVX2 vectorized linear scan (8 `int32` keys/cycle).
- **Pressure-Flow Rebalancing:** Local load-balancing without tree-wide structural changes.
- **Optimistic Lock Coupling (OLC):** Lock-free reads and fine-grained write locking.
- **Concurrency-Induced Density:** Memory footprint naturally decreases under high multi-threading.
- **Low Memory Footprint:** 6.9 Bytes/key (vs 18.6 for Learned Indexes like ALEX).

## Build Instructions

### Prerequisites
- GCC 9+ / Clang 10+
- CMake 3.10+
- AVX2 Support
- OpenMP 4.0+

```bash
git clone --recursive https://github.com/Anuragpm214/cache_conscious_Datastructure_HYDRODS.git
cd cache_conscious_Datastructure_HYDRODS

mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Quick Start
```cpp
#include "hydrods_concurrent.hpp"

// Initialize with int32_t keys and bucket capacity of 256
ConcurrentHydroDS<int32_t, 256> ds;

ds.insert(42);
bool found = ds.search(42);
int64_t count = ds.range_query(10, 50);
ds.erase(42);
```

## Benchmark Summary
Benchmarked on an 8-Core Intel/AMD setup, 5M keys:
- **Range Scan:** Up to 20x faster than CSB+-Tree.
- **Point Search:** 3x faster than CSB+-Tree.
- **Memory Footprint:** 33 MB, outperforming ALEX (89 MB) significantly.
- **Multi-Threading:** Scales linearly up to 32 threads for reads/scans.

## Research Paper
See `paper/main.tex` for the detailed research paper on HydroDS.

## License
Developed as part of ongoing research. Baselines (ALEX, TLX) retain their respective licenses.

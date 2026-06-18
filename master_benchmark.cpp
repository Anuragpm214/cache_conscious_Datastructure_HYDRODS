#include "hydrods_concurrent.hpp"
#include <tlx/container/btree_multiset.hpp>
#include <alex.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <string>
#include <cstdio>
#include <cstring>
#include <set>

using namespace std;
using clk = chrono::high_resolution_clock;

// OS-level accurate physical memory usage
size_t get_memory_usage_mb() {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;
    char line[128];
    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            size_t val;
            sscanf(line, "VmRSS: %zu kB", &val);
            fclose(file);
            return val / 1024;
        }
    }
    fclose(file);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: taskset -c 0 perf stat -e cache-misses,L1-dcache-load-misses ./master_benchmark [hydrods|alex|tlx_btree|std_set]\n";
        return 1;
    }
    string mode = argv[1];
    
    // Benchmark sizing
    const int N = 5000000;
    const int SEARCH_N = 500000;
    
    // Range query sizes
    const int RANGE_LEN_SMALL = 100;
    const int RANGE_COUNT_SMALL = 50000;
    const int RANGE_LEN_MEDIUM = 10000;
    const int RANGE_COUNT_MEDIUM = 5000;
    const int RANGE_LEN_LARGE = 500000;
    const int RANGE_COUNT_LARGE = 50;
    
    cout << "--- Initializing Workload Data ---" << endl;
    vector<int32_t> data(N);
    iota(data.begin(), data.end(), 0);
    mt19937 rng(42);
    shuffle(data.begin(), data.end(), rng);
    
    uniform_int_distribution<size_t> dist(0, data.size() - 1);
    
    vector<int32_t> search_queries(SEARCH_N);
    for (int i = 0; i < SEARCH_N; ++i) search_queries[i] = data[dist(rng)];
    
    uniform_int_distribution<int32_t> range_dist_s(0, N - RANGE_LEN_SMALL - 1);
    vector<int32_t> range_starts_small(RANGE_COUNT_SMALL);
    for (int i = 0; i < RANGE_COUNT_SMALL; ++i) range_starts_small[i] = range_dist_s(rng);

    uniform_int_distribution<int32_t> range_dist_m(0, N - RANGE_LEN_MEDIUM - 1);
    vector<int32_t> range_starts_medium(RANGE_COUNT_MEDIUM);
    for (int i = 0; i < RANGE_COUNT_MEDIUM; ++i) range_starts_medium[i] = range_dist_m(rng);

    uniform_int_distribution<int32_t> range_dist_l(0, N - RANGE_LEN_LARGE - 1);
    vector<int32_t> range_starts_large(RANGE_COUNT_LARGE);
    for (int i = 0; i < RANGE_COUNT_LARGE; ++i) range_starts_large[i] = range_dist_l(rng);

    vector<int32_t> delete_queries(SEARCH_N);
    for (int i = 0; i < SEARCH_N; ++i) delete_queries[i] = data[dist(rng)];

    size_t mem_before = get_memory_usage_mb();

    cout << "--- Starting Execution for: " << mode << " ---" << endl;

    if (mode == "hydrods") {
        ConcurrentHydroDS<int32_t, 256> h; 
        
        auto t1 = clk::now();
        for (int x : data) h.insert(x);
        auto t2 = clk::now();
        size_t mem_after = get_memory_usage_mb();

        auto s1 = clk::now();
        uint64_t search_chk = 0;
        for (int q : search_queries) search_chk += h.search(q);
        auto s2 = clk::now();
        
        auto rs1 = clk::now();
        uint64_t range_chk_s = 0;
        for (int start : range_starts_small) range_chk_s += h.range_query(start, start + RANGE_LEN_SMALL);
        auto rs2 = clk::now();

        auto rm1 = clk::now();
        uint64_t range_chk_m = 0;
        for (int start : range_starts_medium) range_chk_m += h.range_query(start, start + RANGE_LEN_MEDIUM);
        auto rm2 = clk::now();

        auto rl1 = clk::now();
        uint64_t range_chk_l = 0;
        for (int start : range_starts_large) range_chk_l += h.range_query(start, start + RANGE_LEN_LARGE);
        auto rl2 = clk::now();
        
        auto d1 = clk::now();
        uint64_t del_chk = 0;
        for (int q : delete_queries) del_chk += h.erase(q);
        auto d2 = clk::now();

        cout << "[Perf] Insert 5M:            " << chrono::duration<double>(t2 - t1).count() << " s\n";
        cout << "[Perf] Search 500k:          " << chrono::duration<double>(s2 - s1).count() << " s\n";
        cout << "[Perf] Range (Small) 50k:    " << chrono::duration<double>(rs2 - rs1).count() << " s\n";
        cout << "[Perf] Range (Medium) 5k:    " << chrono::duration<double>(rm2 - rm1).count() << " s\n";
        cout << "[Perf] Range (Large) 50:     " << chrono::duration<double>(rl2 - rl1).count() << " s\n";
        cout << "[Perf] Delete 500k:          " << chrono::duration<double>(d2 - d1).count() << " s\n";
        cout << "[Mem]  Physical RAM:         " << (mem_after - mem_before) << " MB\n";
        cout << "[Mem]  Bytes per Key:        " << (double)((mem_after - mem_before) * 1024 * 1024) / N << " bytes\n";
        cout << "[CHK]  " << search_chk << "|" << range_chk_s << "|" << range_chk_m << "|" << range_chk_l << "|" << del_chk << "\n";
        
    } else if (mode == "tlx_btree" || mode == "csb_tree") {
        tlx::btree_multiset<int32_t> b;
        
        auto t1 = clk::now();
        for (int x : data) b.insert(x);
        auto t2 = clk::now();
        size_t mem_after = get_memory_usage_mb();

        auto s1 = clk::now();
        uint64_t search_chk = 0;
        for (int q : search_queries) search_chk += (b.find(q) != b.end());
        auto s2 = clk::now();
        
        auto rs1 = clk::now();
        uint64_t range_chk_s = 0;
        for (int start : range_starts_small) {
            auto it_start = b.lower_bound(start); auto it_end = b.upper_bound(start + RANGE_LEN_SMALL);
            int64_t c = 0; for (auto it = it_start; it != it_end; ++it) c++;
            range_chk_s += c;
        }
        auto rs2 = clk::now();

        auto rm1 = clk::now();
        uint64_t range_chk_m = 0;
        for (int start : range_starts_medium) {
            auto it_start = b.lower_bound(start); auto it_end = b.upper_bound(start + RANGE_LEN_MEDIUM);
            int64_t c = 0; for (auto it = it_start; it != it_end; ++it) c++;
            range_chk_m += c;
        }
        auto rm2 = clk::now();

        auto rl1 = clk::now();
        uint64_t range_chk_l = 0;
        for (int start : range_starts_large) {
            auto it_start = b.lower_bound(start); auto it_end = b.upper_bound(start + RANGE_LEN_LARGE);
            int64_t c = 0; for (auto it = it_start; it != it_end; ++it) c++;
            range_chk_l += c;
        }
        auto rl2 = clk::now();

        auto d1 = clk::now();
        uint64_t del_chk = 0;
        for (int q : delete_queries) {
            auto it = b.find(q);
            if (it != b.end()) { b.erase(it); del_chk++; }
        }
        auto d2 = clk::now();
        
        cout << "[Perf] Insert 5M:            " << chrono::duration<double>(t2 - t1).count() << " s\n";
        cout << "[Perf] Search 500k:          " << chrono::duration<double>(s2 - s1).count() << " s\n";
        cout << "[Perf] Range (Small) 50k:    " << chrono::duration<double>(rs2 - rs1).count() << " s\n";
        cout << "[Perf] Range (Medium) 5k:    " << chrono::duration<double>(rm2 - rm1).count() << " s\n";
        cout << "[Perf] Range (Large) 50:     " << chrono::duration<double>(rl2 - rl1).count() << " s\n";
        cout << "[Perf] Delete 500k:          " << chrono::duration<double>(d2 - d1).count() << " s\n";
        cout << "[Mem]  Physical RAM:         " << (mem_after - mem_before) << " MB\n";
        cout << "[Mem]  Bytes per Key:        " << (double)((mem_after - mem_before) * 1024 * 1024) / N << " bytes\n";
        cout << "[CHK]  " << search_chk << "|" << range_chk_s << "|" << range_chk_m << "|" << range_chk_l << "|" << del_chk << "\n";

    } else if (mode == "alex") {
        alex::Alex<int32_t, int32_t> a;
        
        auto t1 = clk::now();
        for (int x : data) a.insert(x, 0);
        auto t2 = clk::now();
        size_t mem_after = get_memory_usage_mb();

        auto s1 = clk::now();
        uint64_t search_chk = 0;
        for (int q : search_queries) {
            if (a.get_payload(q) != nullptr) search_chk++;
        }
        auto s2 = clk::now();
        
        auto rs1 = clk::now();
        uint64_t range_chk_s = 0;
        for (int start : range_starts_small) {
            auto it = a.lower_bound(start); auto it_end = a.upper_bound(start + RANGE_LEN_SMALL);
            int64_t c = 0; for (; it != it_end; ++it) c++;
            range_chk_s += c;
        }
        auto rs2 = clk::now();

        auto rm1 = clk::now();
        uint64_t range_chk_m = 0;
        for (int start : range_starts_medium) {
            auto it = a.lower_bound(start); auto it_end = a.upper_bound(start + RANGE_LEN_MEDIUM);
            int64_t c = 0; for (; it != it_end; ++it) c++;
            range_chk_m += c;
        }
        auto rm2 = clk::now();

        auto rl1 = clk::now();
        uint64_t range_chk_l = 0;
        for (int start : range_starts_large) {
            auto it = a.lower_bound(start); auto it_end = a.upper_bound(start + RANGE_LEN_LARGE);
            int64_t c = 0; for (; it != it_end; ++it) c++;
            range_chk_l += c;
        }
        auto rl2 = clk::now();

        auto d1 = clk::now();
        uint64_t del_chk = 0;
        for (int q : delete_queries) {
            del_chk += a.erase(q);
        }
        auto d2 = clk::now();
        
        cout << "[Perf] Insert 5M:            " << chrono::duration<double>(t2 - t1).count() << " s\n";
        cout << "[Perf] Search 500k:          " << chrono::duration<double>(s2 - s1).count() << " s\n";
        cout << "[Perf] Range (Small) 50k:    " << chrono::duration<double>(rs2 - rs1).count() << " s\n";
        cout << "[Perf] Range (Medium) 5k:    " << chrono::duration<double>(rm2 - rm1).count() << " s\n";
        cout << "[Perf] Range (Large) 50:     " << chrono::duration<double>(rl2 - rl1).count() << " s\n";
        cout << "[Perf] Delete 500k:          " << chrono::duration<double>(d2 - d1).count() << " s\n";
        cout << "[Mem]  Physical RAM:         " << (mem_after - mem_before) << " MB\n";
        cout << "[Mem]  Bytes per Key:        " << (double)((mem_after - mem_before) * 1024 * 1024) / N << " bytes\n";
        cout << "[CHK]  " << search_chk << "|" << range_chk_s << "|" << del_chk << "\n";
        
    } else {
        cerr << "Unknown structure: " << mode << endl;
        return 1;
    }
    
    return 0;
}

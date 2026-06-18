#include "hydrods_concurrent.hpp"
#include <tlx/container/btree_multiset.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <chrono>
#include <numeric>
#include <string>
#include <cstdio>
#include <cstring>

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
    if (argc != 2) {
        cerr << "Usage: ./benchmark [hydrods|csb_btree]\n";
        return 1;
    }
    string mode = argv[1];
    
    const int N = 5000000;
    const int SEARCH_N = 500000;
    
    // Range query sizes
    const int RANGE_LEN_SMALL = 100;
    const int RANGE_COUNT_SMALL = 50000;

    const int RANGE_LEN_MEDIUM = 10000;
    const int RANGE_COUNT_MEDIUM = 5000;

    const int RANGE_LEN_LARGE = 500000;
    const int RANGE_COUNT_LARGE = 50;
    
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

    if (mode == "hydrods") {
        cout << "--- Benchmarking ConcurrentHydroDS ---" << endl;
        ConcurrentHydroDS<int32_t, 128> h; 
        
        auto t1 = clk::now();
        for (int x : data) h.insert(x);
        auto t2 = clk::now();
        
        size_t mem_after = get_memory_usage_mb();

        auto s1 = clk::now();
        uint64_t search_checksum = 0;
        for (int q : search_queries) search_checksum += h.search(q);
        auto s2 = clk::now();
        
        // Small Ranges
        auto rs1 = clk::now();
        uint64_t range_chk_small = 0;
        for (int start : range_starts_small) range_chk_small += h.range_query(start, start + RANGE_LEN_SMALL);
        auto rs2 = clk::now();

        // Medium Ranges
        auto rm1 = clk::now();
        uint64_t range_chk_medium = 0;
        for (int start : range_starts_medium) range_chk_medium += h.range_query(start, start + RANGE_LEN_MEDIUM);
        auto rm2 = clk::now();

        // Large Ranges
        auto rl1 = clk::now();
        uint64_t range_chk_large = 0;
        for (int start : range_starts_large) range_chk_large += h.range_query(start, start + RANGE_LEN_LARGE);
        auto rl2 = clk::now();
        
        auto d1 = clk::now();
        uint64_t delete_checksum = 0;
        for (int q : delete_queries) delete_checksum += h.erase(q);
        auto d2 = clk::now();

        cout << "Insert 5M:            " << chrono::duration<double>(t2 - t1).count() << " s\n";
        cout << "Search 500k:          " << chrono::duration<double>(s2 - s1).count() << " s\n";
        cout << "Range (Small) 50k:    " << chrono::duration<double>(rs2 - rs1).count() << " s  [chk: " << range_chk_small << "]\n";
        cout << "Range (Medium) 5k:    " << chrono::duration<double>(rm2 - rm1).count() << " s  [chk: " << range_chk_medium << "]\n";
        cout << "Range (Large) 50:     " << chrono::duration<double>(rl2 - rl1).count() << " s  [chk: " << range_chk_large << "]\n";
        cout << "Delete 500k:          " << chrono::duration<double>(d2 - d1).count() << " s\n";
        
        size_t memory_used = mem_after - mem_before;
        cout << "Physical Memory Used: " << memory_used << " MB\n";
        
    } else if (mode == "csb_btree") {
        cout << "--- Benchmarking CSB+-Tree (tlx::btree) ---" << endl;
        tlx::btree_multiset<int32_t> b;
        
        auto t1 = clk::now();
        for (int x : data) b.insert(x);
        auto t2 = clk::now();
        
        size_t mem_after = get_memory_usage_mb();

        auto s1 = clk::now();
        uint64_t search_checksum = 0;
        for (int q : search_queries) search_checksum += (b.find(q) != b.end());
        auto s2 = clk::now();
        
        // Small Ranges
        auto rs1 = clk::now();
        uint64_t range_chk_small = 0;
        for (int start : range_starts_small) {
            auto it_start = b.lower_bound(start);
            auto it_end = b.upper_bound(start + RANGE_LEN_SMALL);
            int64_t c = 0; for (auto it = it_start; it != it_end; ++it) c++;
            range_chk_small += c;
        }
        auto rs2 = clk::now();

        // Medium Ranges
        auto rm1 = clk::now();
        uint64_t range_chk_medium = 0;
        for (int start : range_starts_medium) {
            auto it_start = b.lower_bound(start);
            auto it_end = b.upper_bound(start + RANGE_LEN_MEDIUM);
            int64_t c = 0; for (auto it = it_start; it != it_end; ++it) c++;
            range_chk_medium += c;
        }
        auto rm2 = clk::now();

        // Large Ranges
        auto rl1 = clk::now();
        uint64_t range_chk_large = 0;
        for (int start : range_starts_large) {
            auto it_start = b.lower_bound(start);
            auto it_end = b.upper_bound(start + RANGE_LEN_LARGE);
            int64_t c = 0; for (auto it = it_start; it != it_end; ++it) c++;
            range_chk_large += c;
        }
        auto rl2 = clk::now();

        auto d1 = clk::now();
        uint64_t delete_checksum = 0;
        for (int q : delete_queries) {
            auto it = b.find(q);
            if (it != b.end()) {
                b.erase(it);
                delete_checksum++;
            }
        }
        auto d2 = clk::now();
        
        cout << "Insert 5M:            " << chrono::duration<double>(t2 - t1).count() << " s\n";
        cout << "Search 500k:          " << chrono::duration<double>(s2 - s1).count() << " s\n";
        cout << "Range (Small) 50k:    " << chrono::duration<double>(rs2 - rs1).count() << " s  [chk: " << range_chk_small << "]\n";
        cout << "Range (Medium) 5k:    " << chrono::duration<double>(rm2 - rm1).count() << " s  [chk: " << range_chk_medium << "]\n";
        cout << "Range (Large) 50:     " << chrono::duration<double>(rl2 - rl1).count() << " s  [chk: " << range_chk_large << "]\n";
        cout << "Delete 500k:          " << chrono::duration<double>(d2 - d1).count() << " s\n";
        
        size_t memory_used = mem_after - mem_before;
        cout << "Physical Memory Used: " << memory_used << " MB\n";
    }
    
    return 0;
}

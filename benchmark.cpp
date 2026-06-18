#include "hydrods_concurrent.hpp"
#include <tlx/container/btree_multiset.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <chrono>
#include <numeric>
#include <string>

using namespace std;
using clk = chrono::high_resolution_clock;

// Command line arg: ./benchmark [hydrods | btree | rbtree]
int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: ./benchmark [hydrods|btree|rbtree]\n";
        return 1;
    }
    string mode = argv[1];
    
    const int N = 5000000;
    const int SEARCH_N = 500000;
    const int RANGE_N = 50000;
    const int RANGE_LEN = 10000;
    
    vector<int32_t> data(N);
    iota(data.begin(), data.end(), 0);
    mt19937 rng(42);
    shuffle(data.begin(), data.end(), rng);
    
    uniform_int_distribution<size_t> dist(0, data.size() - 1);
    uniform_int_distribution<int32_t> range_dist(0, N - RANGE_LEN - 1);
    
    // Generate queries ahead of time so RNG overhead is excluded
    vector<int32_t> search_queries(SEARCH_N);
    for (int i = 0; i < SEARCH_N; ++i) search_queries[i] = data[dist(rng)];
    
    vector<int32_t> range_starts(RANGE_N);
    for (int i = 0; i < RANGE_N; ++i) range_starts[i] = range_dist(rng);

    if (mode == "hydrods") {
        cout << "--- Benchmarking ConcurrentHydroDS ---" << endl;
        ConcurrentHydroDS<int32_t, 128> h; // 128 is sweet spot for cache lines
        
        auto t1 = clk::now();
        for (int x : data) h.insert(x);
        auto t2 = clk::now();
        
        auto s1 = clk::now();
        volatile int s_cnt = 0;
        for (int q : search_queries) s_cnt += h.search(q);
        auto s2 = clk::now();
        
        auto r1 = clk::now();
        volatile int64_t r_cnt = 0;
        for (int start : range_starts) r_cnt += h.range_query(start, start + RANGE_LEN);
        auto r2 = clk::now();
        
        cout << "Insert 5M: " << chrono::duration<double>(t2 - t1).count() << " s\n";
        cout << "Search 500k: " << chrono::duration<double>(s2 - s1).count() << " s\n";
        cout << "Range 50k: " << chrono::duration<double>(r2 - r1).count() << " s\n";
        h.print_stats();
        
    } else if (mode == "btree") {
        cout << "--- Benchmarking tlx::btree_multiset ---" << endl;
        tlx::btree_multiset<int32_t> b;
        
        auto t1 = clk::now();
        for (int x : data) b.insert(x);
        auto t2 = clk::now();
        
        auto s1 = clk::now();
        volatile int s_cnt = 0;
        for (int q : search_queries) s_cnt += (b.find(q) != b.end());
        auto s2 = clk::now();
        
        auto r1 = clk::now();
        volatile int64_t r_cnt = 0;
        for (int start : range_starts) {
            auto it_start = b.lower_bound(start);
            auto it_end = b.upper_bound(start + RANGE_LEN);
            // Count elements (tlx iterators don't have O(1) distance)
            int64_t c = 0;
            for (auto it = it_start; it != it_end; ++it) c++;
            r_cnt += c;
        }
        auto r2 = clk::now();
        
        cout << "Insert 5M: " << chrono::duration<double>(t2 - t1).count() << " s\n";
        cout << "Search 500k: " << chrono::duration<double>(s2 - s1).count() << " s\n";
        cout << "Range 50k: " << chrono::duration<double>(r2 - r1).count() << " s\n";
        
    } else if (mode == "rbtree") {
        cout << "--- Benchmarking std::multiset ---" << endl;
        std::multiset<int32_t> rb;
        
        auto t1 = clk::now();
        for (int x : data) rb.insert(x);
        auto t2 = clk::now();
        
        auto s1 = clk::now();
        volatile int s_cnt = 0;
        for (int q : search_queries) s_cnt += (rb.find(q) != rb.end());
        auto s2 = clk::now();
        
        auto r1 = clk::now();
        volatile int64_t r_cnt = 0;
        for (int start : range_starts) {
            auto it_start = rb.lower_bound(start);
            auto it_end = rb.upper_bound(start + RANGE_LEN);
            int64_t c = 0;
            for (auto it = it_start; it != it_end; ++it) c++;
            r_cnt += c;
        }
        auto r2 = clk::now();
        
        cout << "Insert 5M: " << chrono::duration<double>(t2 - t1).count() << " s\n";
        cout << "Search 500k: " << chrono::duration<double>(s2 - s1).count() << " s\n";
        cout << "Range 50k: " << chrono::duration<double>(r2 - r1).count() << " s\n";
    }
    
    return 0;
}

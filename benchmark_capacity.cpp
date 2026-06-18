#include "hydrods_concurrent.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <algorithm>

using namespace std;
using clk = chrono::high_resolution_clock;

template <int C>
void test_capacity(const vector<int32_t>& data, const vector<int32_t>& search_queries) {
    ConcurrentHydroDS<int32_t, C> h;
    
    // Insert benchmark
    auto t1 = clk::now();
    for (int x : data) h.insert(x);
    auto t2 = clk::now();
    
    // Search benchmark
    auto s1 = clk::now();
    uint64_t sum = 0;
    for (int q : search_queries) sum += h.search(q);
    auto s2 = clk::now();
    
    double ins_time = chrono::duration<double>(t2 - t1).count();
    double search_time = chrono::duration<double>(s2 - s1).count();
    
    double ins_mops = (data.size() / 1000000.0) / ins_time;
    double search_mops = (search_queries.size() / 1000000.0) / search_time;
    
    cout << setw(8) << C 
         << setw(20) << fixed << setprecision(2) << ins_mops 
         << setw(20) << fixed << setprecision(2) << search_mops 
         << "        [chk: " << sum << "]" << endl;
}

int main() {
    const int N = 2000000;
    const int SEARCH_N = 2000000;
    
    vector<int32_t> data(N);
    iota(data.begin(), data.end(), 0);
    mt19937 rng(42);
    shuffle(data.begin(), data.end(), rng);
    
    uniform_int_distribution<size_t> dist(0, data.size() - 1);
    vector<int32_t> search_queries(SEARCH_N);
    for (int i = 0; i < SEARCH_N; ++i) search_queries[i] = data[dist(rng)];
    
    cout << "=== HydroDS Capacity Sensitivity Analysis ===" << endl;
    cout << "Testing Insert and Search throughput for various Bucket Capacities." << endl;
    cout << "Keys: " << N << " | Queries: " << SEARCH_N << "\n" << endl;
    
    cout << setw(8) << "Capacity" 
         << setw(20) << "Insert (MOps/s)" 
         << setw(20) << "Search (MOps/s)" << endl;
    cout << string(60, '-') << endl;
    
    test_capacity<16>(data, search_queries);
    test_capacity<32>(data, search_queries);
    test_capacity<64>(data, search_queries);
    test_capacity<128>(data, search_queries);
    test_capacity<256>(data, search_queries);
    test_capacity<512>(data, search_queries);
    test_capacity<1024>(data, search_queries);
    
    return 0;
}

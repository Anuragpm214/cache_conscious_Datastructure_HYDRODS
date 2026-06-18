#include "hydrods.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <algorithm>
#include <chrono>
#include <set>

using namespace std;

void test_edge_cases() {
    cout << "--- Testing Edge Cases ---" << endl;
    HydroDS<int32_t, 64> h;

    // 1. Empty struct
    assert(h.size() == 0);
    assert(!h.search(10));
    assert(!h.erase(10));
    assert(h.range_query(0, 100) == 0);

    // 2. Single element
    h.insert(42);
    assert(h.size() == 1);
    assert(h.search(42));
    assert(!h.search(43));
    assert(h.range_query(40, 50) == 1);
    
    // 3. Duplicate insertion
    h.insert(42);
    assert(h.size() == 2);
    assert(h.range_query(40, 50) == 2);
    
    // 4. Erase duplicates
    assert(h.erase(42));
    assert(h.size() == 1);
    assert(h.search(42)); // One still remains
    assert(h.erase(42));
    assert(h.size() == 0);
    assert(!h.search(42));

    // 5. Sequential Insert (triggers fast-append)
    for (int i = 0; i < 100; ++i) {
        h.insert(i);
    }
    assert(h.size() == 100);
    for (int i = 0; i < 100; ++i) {
        assert(h.search(i));
    }

    // 6. Reverse Sequential Insert (triggers fast-prepend)
    HydroDS<int32_t, 64> h_rev;
    for (int i = 99; i >= 0; --i) {
        h_rev.insert(i);
    }
    assert(h_rev.size() == 100);
    for (int i = 0; i < 100; ++i) {
        assert(h_rev.search(i));
    }

    // 7. Range Queries and Collect
    auto res = h.range_collect(20, 30);
    assert(res.size() == 11);
    for (int i = 0; i <= 10; ++i) {
        assert(res[i] == 20 + i);
    }
    
    // Out of bounds range
    assert(h.range_query(-10, -1) == 0);
    assert(h.range_query(200, 300) == 0);

    // 8. Delete all
    for (int i = 0; i < 100; ++i) {
        assert(h.erase(i));
    }
    assert(h.size() == 0);
    assert(h.empty());
    
    cout << "Edge cases passed!" << endl;
}

void test_randomized_correctness() {
    cout << "--- Testing Randomized Correctness ---" << endl;
    
    HydroDS<int32_t, 256> h;
    multiset<int32_t> ref;
    
    mt19937 rng(1337);
    uniform_int_distribution<int32_t> dist(1, 100000);
    
    const int N = 20000;
    
    // Insertions
    for (int i = 0; i < N; ++i) {
        int32_t val = dist(rng);
        h.insert(val);
        ref.insert(val);
    }
    
    assert(h.size() == ref.size());
    assert(h.verify_integrity());
    
    // Lookups
    for (int i = 0; i < 5000; ++i) {
        int32_t val = dist(rng);
        bool in_h = h.search(val);
        bool in_ref = ref.find(val) != ref.end();
        assert(in_h == in_ref);
    }
    
    // Deletions
    int erase_count = 0;
    for (int i = 0; i < 5000; ++i) {
        int32_t val = dist(rng);
        
        auto it = ref.find(val);
        bool in_ref = (it != ref.end());
        
        bool erased_h = h.erase(val);
        assert(erased_h == in_ref);
        
        if (in_ref) {
            ref.erase(it);
            erase_count++;
        }
    }
    
    assert(h.size() == ref.size());
    assert(h.verify_integrity());
    
    // Range queries
    for (int i = 0; i < 1000; ++i) {
        int32_t lo = dist(rng);
        int32_t hi = lo + dist(rng) % 1000;
        
        int h_count = h.range_query(lo, hi);
        
        auto it_lo = ref.lower_bound(lo);
        auto it_hi = ref.upper_bound(hi);
        int ref_count = distance(it_lo, it_hi);
        
        assert(h_count == ref_count);
    }
    
    h.print_stats();
    cout << "Randomized correctness passed!" << endl;
}

void test_performance_vs_original() {
    cout << "--- Micro-benchmark: Phase 1 ---" << endl;
    
    const int N = 5000000;
    vector<int32_t> data(N);
    iota(data.begin(), data.end(), 0);
    
    mt19937 rng(42);
    shuffle(data.begin(), data.end(), rng);
    
    HydroDS<int32_t, 1024> h;
    
    auto t1 = chrono::high_resolution_clock::now();
    for (int x : data) {
        h.insert(x);
    }
    auto t2 = chrono::high_resolution_clock::now();
    
    uniform_int_distribution<size_t> dist(0, data.size() - 1);
    
    auto s1 = chrono::high_resolution_clock::now();
    volatile int cnt = 0;
    for (size_t i = 0; i < data.size() / 10; ++i) {
        cnt += h.search(data[dist(rng)]);
    }
    auto s2 = chrono::high_resolution_clock::now();
    
    auto d1 = chrono::high_resolution_clock::now();
    volatile int d_cnt = 0;
    for (size_t i = 0; i < data.size() / 10; ++i) {
        d_cnt += h.erase(data[dist(rng)]);
    }
    auto d2 = chrono::high_resolution_clock::now();
    
    cout << "Elements: " << N << endl;
    cout << "Insert 5M: " << chrono::duration<double>(t2 - t1).count() << " s\n";
    cout << "Search 500k: " << chrono::duration<double>(s2 - s1).count() << " s (Found: " << cnt << ")\n";
    cout << "Erase 500k: " << chrono::duration<double>(d2 - d1).count() << " s (Erased: " << d_cnt << ")\n";
    h.print_stats();
}

int main() {
    cout << "Starting HydroDS Phase 1 Tests..." << endl;
    test_edge_cases();
    test_randomized_correctness();
    test_performance_vs_original();
    cout << "All tests completed successfully!" << endl;
    return 0;
}

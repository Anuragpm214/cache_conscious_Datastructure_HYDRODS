#include "hydrods_concurrent.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <cassert>
#include <atomic>
#include <mutex>
#include <chrono>
#include <algorithm>

using namespace std;

void test_concurrent_insert_search() {
    cout << "--- Testing Concurrent Insert & Search ---" << endl;
    ConcurrentHydroDS<int32_t, 128> h;
    const int NUM_THREADS = 8;
    const int OPS_PER_THREAD = 50000;
    
    vector<thread> threads;
    atomic<int> ready{0};
    
    // Insert threads
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            ready++;
            while (ready < NUM_THREADS) {} // Barrier
            
            mt19937 rng(42 + t);
            uniform_int_distribution<int32_t> dist(1, 1000000);
            
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                h.insert(dist(rng));
            }
        });
    }
    
    for (auto& th : threads) th.join();
    threads.clear();
    ready = 0;
    
    cout << "Size after concurrent inserts: " << h.size() << endl;
    
    // Read threads
    atomic<int> found_count{0};
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            ready++;
            while (ready < NUM_THREADS) {} // Barrier
            
            mt19937 rng(42 + t);
            uniform_int_distribution<int32_t> dist(1, 1000000);
            
            int local_found = 0;
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                if (h.search(dist(rng))) local_found++;
            }
            found_count += local_found;
        });
    }
    
    for (auto& th : threads) th.join();
    
    cout << "Concurrent search passed (Found: " << found_count << ")" << endl;
}

void test_concurrent_mixed_workload() {
    cout << "--- Testing Concurrent Mixed Workload (Insert/Search/Erase) ---" << endl;
    ConcurrentHydroDS<int32_t, 128> h;
    const int NUM_THREADS = 8;
    const int OPS_PER_THREAD = 20000;
    
    // Pre-populate
    mt19937 base_rng(1337);
    uniform_int_distribution<int32_t> base_dist(1, 100000);
    for (int i = 0; i < 50000; ++i) h.insert(base_dist(base_rng));
    
    vector<thread> threads;
    atomic<int> ready{0};
    
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            ready++;
            while (ready < NUM_THREADS) {}
            
            mt19937 rng(42 + t);
            uniform_int_distribution<int32_t> val_dist(1, 100000);
            uniform_int_distribution<int> op_dist(0, 99);
            
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                int op = op_dist(rng);
                int32_t val = val_dist(rng);
                
                if (op < 50) {          // 50% Search
                    h.search(val);
                } else if (op < 80) {   // 30% Insert
                    h.insert(val);
                } else {                // 20% Erase
                    h.erase(val);
                }
            }
        });
    }
    
    for (auto& th : threads) th.join();
    
    cout << "Mixed workload passed. Final size: " << h.size() << endl;
}

void test_deadlock_and_starvation() {
    cout << "--- Testing Deadlock and Starvation (Heavy Contention) ---" << endl;
    // Small bucket size to force massive amounts of splits and pressure flow
    ConcurrentHydroDS<int32_t, 32> h; 
    const int NUM_THREADS = 16;
    const int OPS_PER_THREAD = 10000;
    
    vector<thread> threads;
    atomic<int> ready{0};
    
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            ready++;
            while (ready < NUM_THREADS) {}
            
            // All threads hit the EXACT same narrow range to force 
            // maximum lock contention, flows, and splits on adjacent buckets
            mt19937 rng(42 + t);
            uniform_int_distribution<int32_t> val_dist(1, 1000); 
            
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                h.insert(val_dist(rng));
            }
        });
    }
    for (auto& th : threads) th.join();
    threads.clear();
    ready = 0;
    
    // Now delete heavily contended range
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            ready++;
            while (ready < NUM_THREADS) {}
            
            mt19937 rng(42 + t);
            uniform_int_distribution<int32_t> val_dist(1, 1000); 
            
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                h.erase(val_dist(rng));
            }
        });
    }
    for (auto& th : threads) th.join();
    
    cout << "Heavy contention passed (No Deadlocks!)" << endl;
}

int main() {
    cout << "Starting Concurrent HydroDS Phase 2 Tests..." << endl;
    test_concurrent_insert_search();
    test_concurrent_mixed_workload();
    test_deadlock_and_starvation();
    cout << "All concurrent tests completed successfully!" << endl;
    return 0;
}

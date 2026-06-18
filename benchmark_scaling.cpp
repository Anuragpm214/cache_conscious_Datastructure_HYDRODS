#include "hydrods_concurrent.hpp"
#include <tlx/container/btree_multiset.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <cmath>
#include <iomanip>
#include <numeric>

using namespace std;
using clk = chrono::high_resolution_clock;

// --- Fast Zipfian approximation by pre-generating probabilities ---
class ZipfianGenerator {
    vector<double> prob;
    int n;
public:
    ZipfianGenerator(int n, double theta) : n(n) {
        prob.resize(n + 1);
        double sum = 0;
        for (int i = 1; i <= n; ++i) sum += 1.0 / pow(i, theta);
        double c = 1.0 / sum;
        sum = 0;
        for (int i = 1; i <= n; ++i) {
            sum += c / pow(i, theta);
            prob[i] = sum;
        }
    }
    
    int next(mt19937& rng) {
        uniform_real_distribution<double> dist(0.0, 1.0);
        double u = dist(rng);
        int lo = 1, hi = n;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (prob[mid] >= u) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }
};

enum Workload { READ_ONLY, READ_HEAVY, BALANCED };

struct GlobalLockBTree {
    tlx::btree_multiset<int32_t> tree;
    mutable shared_mutex mtx;

    void insert(int32_t x) {
        unique_lock<shared_mutex> lock(mtx);
        tree.insert(x);
    }

    bool search(int32_t x) const {
        shared_lock<shared_mutex> lock(mtx);
        return tree.find(x) != tree.end();
    }
};

void run_benchmark(int num_threads, Workload wl, bool is_zipfian, const string& ds_type) {
    const int INIT_SIZE = 1000000;
    const int NUM_OPS = 2000000; // Operations across ALL threads
    
    // Pre-generate keys
    vector<int32_t> init_keys(INIT_SIZE);
    iota(init_keys.begin(), init_keys.end(), 1);
    mt19937 rng(42);
    shuffle(init_keys.begin(), init_keys.end(), rng);
    
    vector<int32_t> op_keys(NUM_OPS);
    if (is_zipfian) {
        ZipfianGenerator zipf(INIT_SIZE, 0.99); // Highly skewed
        for(int i = 0; i < NUM_OPS; ++i) op_keys[i] = zipf.next(rng);
    } else {
        uniform_int_distribution<int32_t> dist(1, INIT_SIZE);
        for(int i = 0; i < NUM_OPS; ++i) op_keys[i] = dist(rng);
    }
    
    vector<bool> is_read(NUM_OPS);
    uniform_real_distribution<double> p_dist(0.0, 1.0);
    double read_ratio = (wl == READ_ONLY) ? 1.0 : (wl == READ_HEAVY) ? 0.95 : 0.50;
    for(int i = 0; i < NUM_OPS; ++i) is_read[i] = (p_dist(rng) < read_ratio);

    ConcurrentHydroDS<int32_t, 128> hydro;
    GlobalLockBTree btree;

    // Initialization (Single Threaded)
    for(int x : init_keys) {
        if (ds_type == "hydrods") hydro.insert(x);
        else btree.insert(x);
    }

    atomic<int> ready{0};
    atomic<uint64_t> total_found{0};
    vector<thread> threads;
    int ops_per_thread = NUM_OPS / num_threads;

    auto t1 = clk::now();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start_idx = t * ops_per_thread;
            int end_idx = start_idx + ops_per_thread;
            uint64_t local_found = 0;

            ready++;
            while (ready < num_threads) {} // Barrier

            if (ds_type == "hydrods") {
                for (int i = start_idx; i < end_idx; ++i) {
                    if (is_read[i]) {
                        local_found += hydro.search(op_keys[i]);
                    } else {
                        hydro.insert(op_keys[i]);
                    }
                }
            } else {
                for (int i = start_idx; i < end_idx; ++i) {
                    if (is_read[i]) {
                        local_found += btree.search(op_keys[i]);
                    } else {
                        btree.insert(op_keys[i]);
                    }
                }
            }
            total_found += local_found;
        });
    }

    for (auto& th : threads) th.join();
    auto t2 = clk::now();

    double sec = chrono::duration<double>(t2 - t1).count();
    double mops = (NUM_OPS / sec) / 1000000.0;
    
    cout << setw(8) << num_threads << " | " << setw(10) << ds_type << " | " 
         << fixed << setprecision(2) << mops << " Mops/s | Found: " << total_found << endl;
}

int main() {
    cout << "=== Phase 3: Thread Scaling Benchmark ===" << endl;
    cout << "Threads  | Structure  | Throughput (Mops/s)" << endl;
    cout << "-------------------------------------------" << endl;

    vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    cout << "\n[Workload C] 100% Read, Uniform Distribution\n";
    for (int t : thread_counts) run_benchmark(t, READ_ONLY, false, "btree");
    for (int t : thread_counts) run_benchmark(t, READ_ONLY, false, "hydrods");

    cout << "\n[Workload B] 95% Read / 5% Insert, Zipfian (Skewed)\n";
    for (int t : thread_counts) run_benchmark(t, READ_HEAVY, true, "btree");
    for (int t : thread_counts) run_benchmark(t, READ_HEAVY, true, "hydrods");

    cout << "\n[Workload A] 50% Read / 50% Insert, Uniform Distribution\n";
    for (int t : thread_counts) run_benchmark(t, BALANCED, false, "btree");
    for (int t : thread_counts) run_benchmark(t, BALANCED, false, "hydrods");

    return 0;
}

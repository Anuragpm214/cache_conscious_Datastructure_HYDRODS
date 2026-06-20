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
#include <omp.h>
#include <shared_mutex>
#include <mutex>
#include <iomanip>

using namespace std;
using clk = chrono::high_resolution_clock;

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
        cerr << "Usage: ./master_benchmark [hydrods|alex|csb_tree]\n";
        return 1;
    }
    string mode = argv[1];
    
    const int N = 5000000;
    const int SEARCH_N = 500000;
    
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

    vector<int> thread_counts = {1};
    
    cout << "--- Thread Scaling Execution for: " << mode << " ---" << endl;
    cout << setw(4) << "Th" << " | "
         << setw(10) << "Ins(s)" << " | "
         << setw(10) << "Srch(s)" << " | "
         << setw(10) << "R_Sml(s)" << " | "
         << setw(10) << "R_Med(s)" << " | "
         << setw(10) << "R_Lrg(s)" << " | "
         << setw(10) << "Del(s)" << " | "
         << "Mem(MB)" << endl;
    cout << string(90, '-') << endl;

    for (int T : thread_counts) {
        omp_set_num_threads(T);
        size_t mem_before = get_memory_usage_mb();
        
        double t_ins=0, t_srch=0, t_rs=0, t_rm=0, t_rl=0, t_del=0;
        size_t mem_after = 0;
        uint64_t chk_s=0, chk_rs=0, chk_rm=0, chk_rl=0, chk_d=0;

        if (mode == "hydrods") {
            ConcurrentHydroDS<int32_t, 512> h; 
            
            auto start = clk::now();
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) h.insert(data[i]);
            t_ins = chrono::duration<double>(clk::now() - start).count();
            mem_after = get_memory_usage_mb();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_s)
            for (int i = 0; i < SEARCH_N; ++i) chk_s += h.search(search_queries[i]);
            t_srch = chrono::duration<double>(clk::now() - start).count();
            
            start = clk::now();
            #pragma omp parallel for reduction(+:chk_rs)
            for (int i = 0; i < RANGE_COUNT_SMALL; ++i) chk_rs += h.range_query(range_starts_small[i], range_starts_small[i] + RANGE_LEN_SMALL);
            t_rs = chrono::duration<double>(clk::now() - start).count();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_rm)
            for (int i = 0; i < RANGE_COUNT_MEDIUM; ++i) chk_rm += h.range_query(range_starts_medium[i], range_starts_medium[i] + RANGE_LEN_MEDIUM);
            t_rm = chrono::duration<double>(clk::now() - start).count();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_rl)
            for (int i = 0; i < RANGE_COUNT_LARGE; ++i) chk_rl += h.range_query(range_starts_large[i], range_starts_large[i] + RANGE_LEN_LARGE);
            t_rl = chrono::duration<double>(clk::now() - start).count();
            
            start = clk::now();
            #pragma omp parallel for reduction(+:chk_d)
            for (int i = 0; i < SEARCH_N; ++i) chk_d += h.erase(delete_queries[i]);
            t_del = chrono::duration<double>(clk::now() - start).count();

        } else if (mode == "tlx_btree" || mode == "csb_tree") {
            tlx::btree_multiset<int32_t> b;
            std::shared_mutex rw_lock; // Global lock since CSB+-Tree is not thread-safe
            
            auto start = clk::now();
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                std::unique_lock<std::shared_mutex> lock(rw_lock);
                b.insert(data[i]);
            }
            t_ins = chrono::duration<double>(clk::now() - start).count();
            mem_after = get_memory_usage_mb();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_s)
            for (int i = 0; i < SEARCH_N; ++i) {
                std::shared_lock<std::shared_mutex> lock(rw_lock);
                chk_s += (b.find(search_queries[i]) != b.end());
            }
            t_srch = chrono::duration<double>(clk::now() - start).count();
            
            start = clk::now();
            #pragma omp parallel for reduction(+:chk_rs)
            for (int i = 0; i < RANGE_COUNT_SMALL; ++i) {
                std::shared_lock<std::shared_mutex> lock(rw_lock);
                auto it_start = b.lower_bound(range_starts_small[i]); auto it_end = b.upper_bound(range_starts_small[i] + RANGE_LEN_SMALL);
                int64_t c = 0; for (auto it = it_start; it != it_end; ++it) c++;
                chk_rs += c;
            }
            t_rs = chrono::duration<double>(clk::now() - start).count();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_rm)
            for (int i = 0; i < RANGE_COUNT_MEDIUM; ++i) {
                std::shared_lock<std::shared_mutex> lock(rw_lock);
                auto it_start = b.lower_bound(range_starts_medium[i]); auto it_end = b.upper_bound(range_starts_medium[i] + RANGE_LEN_MEDIUM);
                int64_t c = 0; for (auto it = it_start; it != it_end; ++it) c++;
                chk_rm += c;
            }
            t_rm = chrono::duration<double>(clk::now() - start).count();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_rl)
            for (int i = 0; i < RANGE_COUNT_LARGE; ++i) {
                std::shared_lock<std::shared_mutex> lock(rw_lock);
                auto it_start = b.lower_bound(range_starts_large[i]); auto it_end = b.upper_bound(range_starts_large[i] + RANGE_LEN_LARGE);
                int64_t c = 0; for (auto it = it_start; it != it_end; ++it) c++;
                chk_rl += c;
            }
            t_rl = chrono::duration<double>(clk::now() - start).count();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_d)
            for (int i = 0; i < SEARCH_N; ++i) {
                std::unique_lock<std::shared_mutex> lock(rw_lock);
                auto it = b.find(delete_queries[i]);
                if (it != b.end()) { b.erase(it); chk_d++; }
            }
            t_del = chrono::duration<double>(clk::now() - start).count();

        } else if (mode == "alex") {
            alex::Alex<int32_t, int32_t> a;
            std::shared_mutex rw_lock; // Global lock since ALEX is not thread-safe
            
            auto start = clk::now();
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                std::unique_lock<std::shared_mutex> lock(rw_lock);
                a.insert(data[i], 0);
            }
            t_ins = chrono::duration<double>(clk::now() - start).count();
            mem_after = get_memory_usage_mb();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_s)
            for (int i = 0; i < SEARCH_N; ++i) {
                std::shared_lock<std::shared_mutex> lock(rw_lock);
                if (a.get_payload(search_queries[i]) != nullptr) chk_s++;
            }
            t_srch = chrono::duration<double>(clk::now() - start).count();
            
            start = clk::now();
            #pragma omp parallel for reduction(+:chk_rs)
            for (int i = 0; i < RANGE_COUNT_SMALL; ++i) {
                std::shared_lock<std::shared_mutex> lock(rw_lock);
                auto it = a.lower_bound(range_starts_small[i]); auto it_end = a.upper_bound(range_starts_small[i] + RANGE_LEN_SMALL);
                int64_t c = 0; for (; it != it_end; ++it) c++;
                chk_rs += c;
            }
            t_rs = chrono::duration<double>(clk::now() - start).count();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_rm)
            for (int i = 0; i < RANGE_COUNT_MEDIUM; ++i) {
                std::shared_lock<std::shared_mutex> lock(rw_lock);
                auto it = a.lower_bound(range_starts_medium[i]); auto it_end = a.upper_bound(range_starts_medium[i] + RANGE_LEN_MEDIUM);
                int64_t c = 0; for (; it != it_end; ++it) c++;
                chk_rm += c;
            }
            t_rm = chrono::duration<double>(clk::now() - start).count();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_rl)
            for (int i = 0; i < RANGE_COUNT_LARGE; ++i) {
                std::shared_lock<std::shared_mutex> lock(rw_lock);
                auto it = a.lower_bound(range_starts_large[i]); auto it_end = a.upper_bound(range_starts_large[i] + RANGE_LEN_LARGE);
                int64_t c = 0; for (; it != it_end; ++it) c++;
                chk_rl += c;
            }
            t_rl = chrono::duration<double>(clk::now() - start).count();

            start = clk::now();
            #pragma omp parallel for reduction(+:chk_d)
            for (int i = 0; i < SEARCH_N; ++i) {
                std::unique_lock<std::shared_mutex> lock(rw_lock);
                chk_d += a.erase(delete_queries[i]);
            }
            t_del = chrono::duration<double>(clk::now() - start).count();
        }

        cout << setw(4) << T << " | "
             << setw(10) << fixed << setprecision(4) << t_ins << " | "
             << setw(10) << fixed << setprecision(4) << t_srch << " | "
             << setw(10) << fixed << setprecision(4) << t_rs << " | "
             << setw(10) << fixed << setprecision(4) << t_rm << " | "
             << setw(10) << fixed << setprecision(4) << t_rl << " | "
             << setw(10) << fixed << setprecision(4) << t_del << " | "
             << (mem_after - mem_before) << " MB" << endl;
             
        // A little anti-DCE hack if needed, or simply let reduction happen.
        // We know compiler won't DCE because we do reduction on chk_* variables.
    }
    return 0;
}

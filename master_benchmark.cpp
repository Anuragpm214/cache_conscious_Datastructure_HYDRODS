#include "engine/hydrods_concurrent.hpp"
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

// -----------------------------------------------------------------------------
// HydroDS Benchmark Template (Needed because C is a template parameter)
// -----------------------------------------------------------------------------
template <int C>
void run_hydrods_benchmark(int T, int N, int SEARCH_N, double eps_high, double eps_low,
                           const vector<int32_t>& data,
                           const vector<int32_t>& search_queries,
                           const vector<int32_t>& delete_queries,
                           const vector<int32_t>& range_starts_small, int RANGE_LEN_SMALL, int RANGE_COUNT_SMALL,
                           const vector<int32_t>& range_starts_medium, int RANGE_LEN_MEDIUM, int RANGE_COUNT_MEDIUM,
                           const vector<int32_t>& range_starts_large, int RANGE_LEN_LARGE, int RANGE_COUNT_LARGE,
                           size_t mem_before) 
{
    double t_ins=0, t_srch=0, t_rs=0, t_rm=0, t_rl=0, t_del=0;
    size_t mem_after = 0;
    uint64_t chk_s=0, chk_rs=0, chk_rm=0, chk_rl=0, chk_d=0;

    ConcurrentHydroDS<int32_t, C> h; 
    h.set_thresholds(eps_high, eps_low);
    
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

    cout << setw(4) << T << " | "
         << setw(10) << fixed << setprecision(4) << t_ins << " | "
         << setw(10) << fixed << setprecision(4) << t_srch << " | "
         << setw(10) << fixed << setprecision(4) << t_rs << " | "
         << setw(10) << fixed << setprecision(4) << t_rm << " | "
         << setw(10) << fixed << setprecision(4) << t_rl << " | "
         << setw(10) << fixed << setprecision(4) << t_del << " | "
         << (mem_after - mem_before) << " MB" << endl;
}

void print_usage() {
    cout << "Usage: ./master_benchmark [options]\n"
         << "Options:\n"
         << "  --mode <str>      Mode: hydrods, alex, csb_tree (default: hydrods)\n"
         << "  --n <int>         Number of elements to insert (default: 5000000)\n"
         << "  --threads <int>   Number of OpenMP threads (default: 1)\n"
         << "  --cap <int>       Bucket capacity for HydroDS: 100, 256, 500, 1024, 2048, 4096 (default: 500)\n"
         << "  --high <float>    EPS_HIGH for HydroDS (default: 0.85)\n"
         << "  --low <float>     EPS_LOW for HydroDS (default: 0.60)\n"
         << "  --help            Show this help message\n";
}

int main(int argc, char** argv) {
    string mode = "hydrods";
    int N = 5000000;
    int num_threads = 1;
    int capacity = 500;
    double eps_high = 0.85;
    double eps_low = 0.60;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--n" && i + 1 < argc) N = stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) num_threads = stoi(argv[++i]);
        else if (arg == "--cap" && i + 1 < argc) capacity = stoi(argv[++i]);
        else if (arg == "--high" && i + 1 < argc) eps_high = stod(argv[++i]);
        else if (arg == "--low" && i + 1 < argc) eps_low = stod(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg.find("--") == string::npos && i == 1) {
            // Support legacy format: `./master_benchmark hydrods`
            mode = arg;
        } else {
            cerr << "Unknown argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }
    
    const int SEARCH_N = N / 10;
    
    const int RANGE_LEN_SMALL = 100;
    const int RANGE_COUNT_SMALL = 50000;
    const int RANGE_LEN_MEDIUM = 10000;
    const int RANGE_COUNT_MEDIUM = 5000;
    const int RANGE_LEN_LARGE = 500000;
    const int RANGE_COUNT_LARGE = 50;
    
    cout << "--- Configuration ---\n"
         << "Mode: " << mode << "\n"
         << "N: " << N << "\n"
         << "Threads: " << num_threads << "\n";
         
    if (mode == "hydrods") {
        cout << "HydroDS Capacity (C): " << capacity << "\n"
             << "HydroDS EPS_HIGH: " << eps_high << "\n"
             << "HydroDS EPS_LOW: " << eps_low << "\n";
    }
    cout << "---------------------\n";
    
    cout << "--- Initializing Workload Data ---" << endl;
    vector<int32_t> data(N);
    iota(data.begin(), data.end(), 0);
    mt19937 rng(42);
    shuffle(data.begin(), data.end(), rng);
    
    uniform_int_distribution<size_t> dist(0, data.size() - 1);
    vector<int32_t> search_queries(SEARCH_N);
    for (int i = 0; i < SEARCH_N; ++i) search_queries[i] = data[dist(rng)];
    
    // Protect against N being too small for range queries
    int max_s = max(1, N - RANGE_LEN_SMALL - 1);
    int max_m = max(1, N - RANGE_LEN_MEDIUM - 1);
    int max_l = max(1, N - RANGE_LEN_LARGE - 1);

    uniform_int_distribution<int32_t> range_dist_s(0, max_s);
    vector<int32_t> range_starts_small(RANGE_COUNT_SMALL);
    for (int i = 0; i < RANGE_COUNT_SMALL; ++i) range_starts_small[i] = range_dist_s(rng);

    uniform_int_distribution<int32_t> range_dist_m(0, max_m);
    vector<int32_t> range_starts_medium(RANGE_COUNT_MEDIUM);
    for (int i = 0; i < RANGE_COUNT_MEDIUM; ++i) range_starts_medium[i] = range_dist_m(rng);

    uniform_int_distribution<int32_t> range_dist_l(0, max_l);
    vector<int32_t> range_starts_large(RANGE_COUNT_LARGE);
    for (int i = 0; i < RANGE_COUNT_LARGE; ++i) range_starts_large[i] = range_dist_l(rng);

    vector<int32_t> delete_queries(SEARCH_N);
    for (int i = 0; i < SEARCH_N; ++i) delete_queries[i] = data[dist(rng)];

    cout << "--- Executing Benchmark for: " << mode << " ---" << endl;
    cout << setw(4) << "Th" << " | "
         << setw(10) << "Ins(s)" << " | "
         << setw(10) << "Srch(s)" << " | "
         << setw(10) << "R_Sml(s)" << " | "
         << setw(10) << "R_Med(s)" << " | "
         << setw(10) << "R_Lrg(s)" << " | "
         << setw(10) << "Del(s)" << " | "
         << "Mem(MB)" << endl;
    cout << string(90, '-') << endl;

    omp_set_num_threads(num_threads);
    size_t mem_before = get_memory_usage_mb();

    double t_ins=0, t_srch=0, t_rs=0, t_rm=0, t_rl=0, t_del=0;
    size_t mem_after = 0;
    uint64_t chk_s=0, chk_rs=0, chk_rm=0, chk_rl=0, chk_d=0;

    if (mode == "hydrods") {
        if (capacity <= 100) run_hydrods_benchmark<100>(num_threads, N, SEARCH_N, eps_high, eps_low, data, search_queries, delete_queries, range_starts_small, RANGE_LEN_SMALL, RANGE_COUNT_SMALL, range_starts_medium, RANGE_LEN_MEDIUM, RANGE_COUNT_MEDIUM, range_starts_large, RANGE_LEN_LARGE, RANGE_COUNT_LARGE, mem_before);
        else if (capacity <= 256) run_hydrods_benchmark<256>(num_threads, N, SEARCH_N, eps_high, eps_low, data, search_queries, delete_queries, range_starts_small, RANGE_LEN_SMALL, RANGE_COUNT_SMALL, range_starts_medium, RANGE_LEN_MEDIUM, RANGE_COUNT_MEDIUM, range_starts_large, RANGE_LEN_LARGE, RANGE_COUNT_LARGE, mem_before);
        else if (capacity <= 500) run_hydrods_benchmark<500>(num_threads, N, SEARCH_N, eps_high, eps_low, data, search_queries, delete_queries, range_starts_small, RANGE_LEN_SMALL, RANGE_COUNT_SMALL, range_starts_medium, RANGE_LEN_MEDIUM, RANGE_COUNT_MEDIUM, range_starts_large, RANGE_LEN_LARGE, RANGE_COUNT_LARGE, mem_before);
        else if (capacity <= 1024) run_hydrods_benchmark<1024>(num_threads, N, SEARCH_N, eps_high, eps_low, data, search_queries, delete_queries, range_starts_small, RANGE_LEN_SMALL, RANGE_COUNT_SMALL, range_starts_medium, RANGE_LEN_MEDIUM, RANGE_COUNT_MEDIUM, range_starts_large, RANGE_LEN_LARGE, RANGE_COUNT_LARGE, mem_before);
        else if (capacity <= 2048) run_hydrods_benchmark<2048>(num_threads, N, SEARCH_N, eps_high, eps_low, data, search_queries, delete_queries, range_starts_small, RANGE_LEN_SMALL, RANGE_COUNT_SMALL, range_starts_medium, RANGE_LEN_MEDIUM, RANGE_COUNT_MEDIUM, range_starts_large, RANGE_LEN_LARGE, RANGE_COUNT_LARGE, mem_before);
        else run_hydrods_benchmark<4096>(num_threads, N, SEARCH_N, eps_high, eps_low, data, search_queries, delete_queries, range_starts_small, RANGE_LEN_SMALL, RANGE_COUNT_SMALL, range_starts_medium, RANGE_LEN_MEDIUM, RANGE_COUNT_MEDIUM, range_starts_large, RANGE_LEN_LARGE, RANGE_COUNT_LARGE, mem_before);

    } else if (mode == "tlx_btree" || mode == "csb_tree") {
        tlx::btree_multiset<int32_t> b;
        std::shared_mutex rw_lock;
        
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
        
        cout << setw(4) << num_threads << " | "
             << setw(10) << fixed << setprecision(4) << t_ins << " | "
             << setw(10) << fixed << setprecision(4) << t_srch << " | "
             << setw(10) << fixed << setprecision(4) << t_rs << " | "
             << setw(10) << fixed << setprecision(4) << t_rm << " | "
             << setw(10) << fixed << setprecision(4) << t_rl << " | "
             << setw(10) << fixed << setprecision(4) << t_del << " | "
             << (mem_after - mem_before) << " MB" << endl;

    } else if (mode == "alex") {
        alex::Alex<int32_t, int32_t> a;
        std::shared_mutex rw_lock;
        
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
        
        cout << setw(4) << num_threads << " | "
             << setw(10) << fixed << setprecision(4) << t_ins << " | "
             << setw(10) << fixed << setprecision(4) << t_srch << " | "
             << setw(10) << fixed << setprecision(4) << t_rs << " | "
             << setw(10) << fixed << setprecision(4) << t_rm << " | "
             << setw(10) << fixed << setprecision(4) << t_rl << " | "
             << setw(10) << fixed << setprecision(4) << t_del << " | "
             << (mem_after - mem_before) << " MB" << endl;
    }

    return 0;
}

#include <bits/stdc++.h>
using namespace std;
using clk = chrono::high_resolution_clock;

static std::mt19937 rng(42);

class HydroDS {
    static constexpr int C = 1000;
    static constexpr double EPS_HIGH = 0.85;
    static constexpr double EPS_LOW  = 0.50;

    vector<vector<int>> buckets;
    vector<int> bucket_max;

    inline double pressure(int i) const {
        return double(buckets[i].size()) / C;
    }

    // safe bucket finder
    int find_bucket(int x) const {
        int l = 0, r = (int)bucket_max.size() - 1;
        while (l <= r) {
            int m = (l + r) >> 1;
            if (bucket_max[m] >= x) r = m - 1;
            else l = m + 1;
        }
        if (l >= (int)buckets.size()) l = buckets.size() - 1;
        return l;
    }

    inline void update_index(int i) {
        bucket_max[i] = buckets[i].back();
    }

    // batched flow with ordering preserved
    void flow(int i, int j) {
        double dp = pressure(i) - pressure(j);
        if (dp <= EPS_HIGH) return;

        int k = int(C * (dp - EPS_LOW) / 2.0);
        k = max(1, min(k, (int)buckets[i].size()));

        auto &A = buckets[i];
        auto &B = buckets[j];

        if (i < j) {
            // move largest of A -> front of B
            B.insert(B.begin(), A.end() - k, A.end());
            A.erase(A.end() - k, A.end());
        } else {
            // move smallest of A -> back of B
            B.insert(B.end(), A.begin(), A.begin() + k);
            A.erase(A.begin(), A.begin() + k);
        }

        update_index(i);
        update_index(j);
    }

    // bounded, local stabilization
    void stabilize(int i) {
        bool active = false; //No rebalancing yet
        for (int step = 0; step < 2; ++step) {
            bool moved = false;

            if (i > 0) {
                double dp = pressure(i) - pressure(i - 1);
                if ((!active && dp > EPS_HIGH) || (active && dp >= EPS_LOW)) {
                    active = true; //We are already in a rebalancing phase
                    flow(i, i - 1);
                    moved = true;
                }
            }

            if (i + 1 < (int)buckets.size()) {
                double dp = pressure(i) - pressure(i + 1);
                if ((!active && dp > EPS_HIGH) || (active && dp >= EPS_LOW)) {
                    active = true;
                    flow(i, i + 1);
                    moved = true;
                }
            }

            if (!moved) break;
        }
    }

    void split_bucket(int i) {
        auto &B = buckets[i];
        int mid = B.size() / 2;

        vector<int> right(B.begin() + mid, B.end());
        B.erase(B.begin() + mid, B.end());

        buckets.insert(buckets.begin() + i + 1, move(right));
        bucket_max.insert(bucket_max.begin() + i + 1, buckets[i + 1].back());

        update_index(i);
    }

public:
    HydroDS() {
        buckets.reserve(20000);
        bucket_max.reserve(20000);
    }

    // INSERT
    void insert(int x) {
        if (buckets.empty()) {
            buckets.push_back({x});
            bucket_max.push_back(x);
            return;
        }

        int i = find_bucket(x);
        auto &B = buckets[i];

        // edge-biased insertion
        if (x >= B.back()) {
            B.push_back(x);
        } else if (x <= B.front()) {
            B.insert(B.begin(), x);
        } else {
            B.insert(lower_bound(B.begin(), B.end(), x), x);
        }

        update_index(i);

        if ((int)B.size() > C)
            split_bucket(i);

        stabilize(i);
    }

    // SEARCH
    bool search(int x) const {
        if (buckets.empty()) return false;
        int i = find_bucket(x);
        const auto &B = buckets[i];
        return binary_search(B.begin(), B.end(), x);
    }
    // DELETE
    void erase(int x) {
        if (buckets.empty()) return;

        int i = find_bucket(x);
        if (i < 0 || i >= (int)buckets.size()) return;

        auto &B = buckets[i];
        auto it = lower_bound(B.begin(), B.end(), x);
        if (it == B.end() || *it != x) return;

        B.erase(it);

        // bucket removed entirely
        if (B.empty()) {
            buckets.erase(buckets.begin() + i);
            bucket_max.erase(bucket_max.begin() + i);
            return;
        }

        update_index(i);

        // local underflow stabilization (reverse flow)
        if (i > 0) {
            double dp = pressure(i - 1) - pressure(i);
            if (dp > EPS_HIGH)
                flow(i - 1, i);
        }

        if (i + 1 < (int)buckets.size()) {
            double dp = pressure(i + 1) - pressure(i);
            if (dp > EPS_HIGH)
                flow(i + 1, i);
        }
        
    }
    // RANGE QUERY [L, R]
    int range_query(int L, int R) const {
        if (buckets.empty()) return 0;

        volatile int cnt = 0;
        int i = find_bucket(L);

        for (; i < (int)buckets.size(); ++i) {
            const auto &B = buckets[i];

            if (B.front() > R) break;

            for (int x : B) {
                if (x < L) continue;
                if (x > R) break;
                cnt++;
            }
        }
        return cnt;
    }
};

/* ---------------- BENCHMARK ---------------- */

int main() {
    const int N = 5000000;
    vector<int> data(N);
    iota(data.begin(), data.end(), 0);
    reverse(data.begin(), data.end());
    shuffle(data.begin(), data.end(), mt19937(42));

    HydroDS h;


    //insert
    auto t1 = clk::now();
    for (int x : data) h.insert(x);
    auto t2 = clk::now();


    //search
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);   
    auto s1 = clk::now();
    volatile int cnt = 0;
    for (size_t i = 0; i < data.size() / 10; i++)
        cnt += h.search(data[dist(rng)]);
    auto s2 = clk::now();

    int MAX_KEY   = N - 1;
    int RANGE_LEN = 10000;  
    // RANGE TEST
    std::uniform_int_distribution<int> start_dist(0, MAX_KEY - RANGE_LEN);
    volatile long long rcnt = 0;

    auto r1 = clk::now();
    for (int i = 0; i < 2000; i++) {
        int l = start_dist(rng);
        int r = l + RANGE_LEN;
        rcnt += h.range_query(l, r);
    }
    auto r2 = clk::now();
    cout << "No of element:"<<N<<endl;
    cout << "Hydro range time: "
         << chrono::duration<double>(r2 - r1).count() << " s\n";
    cout << "Hydro insert time: "
         << chrono::duration<double>(t2 - t1).count() << " s\n";
    cout << "Hydro search time: "
         << chrono::duration<double>(s2 - s1).count() << " s\n";
    
}  
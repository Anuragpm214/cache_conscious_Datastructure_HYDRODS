#pragma once
// =============================================================================
//  HydroDS — Phase 2: Concurrency via Optimistic Lock Coupling (OLC)
//
//  Concurrency Model:
//  1. Global Directory Lock: std::shared_mutex protects buckets_ array sizing.
//  2. Bucket OLC: Each bucket has an atomic version counter.
//  3. Reads (Search/Range): Acquire shared directory lock, Optimistically read 
//     bucket without locking. Retry if version changes.
//  4. Writes (Insert/Delete): Acquire shared directory lock, exclusively lock
//     target bucket. If bucket needs split/merge, drop locks and acquire
//     exclusive directory lock.
//  5. Flow (Load Balancing): Locks adjacent buckets in strictly ascending ID
//     order to prevent deadlocks.
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <vector>
#include <iostream>
#include <mutex>

#ifdef __AVX2__
#include <immintrin.h>
// Include SIMD helpers from phase 1
namespace hydro_simd {
inline bool contains_i32(const int32_t* keys, int count, int32_t target) {
    const __m256i t = _mm256_set1_epi32(target);
    int i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys + i));
        __m256i eq = _mm256_cmpeq_epi32(d, t);
        if (_mm256_movemask_epi8(eq)) return true;
        if (keys[i] > target) return false;
    }
    for (; i < count; ++i) {
        if (keys[i] == target) return true;
        if (keys[i] > target)  return false;
    }
    return false;
}
}
#endif

// PAUSE instruction for spinlocks
inline void cpu_pause() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

// -----------------------------------------------------------------------------
// Optimistic Lock Coupling primitives
// -----------------------------------------------------------------------------
struct OptLock {
    std::atomic<uint64_t> version{0};

    bool is_locked(uint64_t v) const { return (v & 1) != 0; }

    uint64_t read_lock() const {
        uint64_t v = version.load(std::memory_order_acquire);
        while (is_locked(v)) {
            cpu_pause();
            v = version.load(std::memory_order_acquire);
        }
        return v;
    }

    bool check_version(uint64_t v) const {
        return version.load(std::memory_order_acquire) == v;
    }

    void write_lock() {
        uint64_t expected = version.load(std::memory_order_relaxed) & ~1ULL;
        while (!version.compare_exchange_weak(expected, expected + 1,
                                              std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
            expected &= ~1ULL; // Reset expected to unlocked state
            cpu_pause();
        }
    }

    void write_unlock() {
        version.fetch_add(1, std::memory_order_release);
    }
};

template <typename Key>
struct AtomicKey {
    std::atomic<Key> val;
    AtomicKey() : val(0) {}
    AtomicKey(Key k) : val(k) {}
    AtomicKey(const AtomicKey& other) : val(other.val.load(std::memory_order_relaxed)) {}
    AtomicKey& operator=(const AtomicKey& other) {
        val.store(other.val.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }
};

template <typename Key = int32_t, int BucketCapacity = 256>
class ConcurrentHydroDS {
    static constexpr int C = BucketCapacity;
    static constexpr double EPS_HIGH = 0.85;
    static constexpr double EPS_LOW  = 0.50;

    struct alignas(64) Bucket {
        OptLock lock;
        int32_t count = 0;
        Key keys[C + 1];

        Key max_key() const { return keys[count - 1]; }
        Key min_key() const { return keys[0]; }
        double pressure() const { return static_cast<double>(count) / C; }
        bool is_empty()    const { return count == 0; }
        bool needs_split() const { return count > C; }

        int lower_bound_pos(Key x) const {
            int lo = 0, hi = count;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (keys[mid] < x) lo = mid + 1;
                else               hi = mid;
            }
            return lo;
        }

        bool contains(Key x) const {
#ifdef __AVX2__
            if constexpr (std::is_same_v<Key, int32_t>) {
                return hydro_simd::contains_i32(keys, count, x);
            } else
#endif
            {
                int pos = lower_bound_pos(x);
                return pos < count && keys[pos] == x;
            }
        }

        void insert_sorted(Key x) {
            if (count == 0 || x >= keys[count - 1]) {
                keys[count++] = x;
                return;
            }
            if (x <= keys[0]) {
                std::memmove(&keys[1], &keys[0], count * sizeof(Key));
                keys[0] = x;
                ++count;
                return;
            }
            int pos = lower_bound_pos(x);
            std::memmove(&keys[pos + 1], &keys[pos], (count - pos) * sizeof(Key));
            keys[pos] = x;
            ++count;
        }

        void remove_at(int pos) {
            if (pos < count - 1) {
                std::memmove(&keys[pos], &keys[pos + 1], (count - pos - 1) * sizeof(Key));
            }
            --count;
        }
    };

    std::vector<Bucket*> buckets_;
    std::vector<AtomicKey<Key>> bucket_max_; // Atomic for concurrent binary search
    std::atomic<uint64_t> dir_seq_{0};       // Sequence lock for wait-free directory reads
    std::mutex split_lock_;                  // Exclusive lock for splits
    std::atomic<size_t> total_size_{0};

    static Bucket* alloc_bucket() {
        void* mem = std::aligned_alloc(alignof(Bucket), sizeof(Bucket));
        if (!mem) throw std::bad_alloc();
        return new (mem) Bucket();
    }

    static void free_bucket(Bucket* b) {
        if (b) {
            b->~Bucket();
            std::free(b);
        }
    }

    int find_bucket(Key x) const {
        while (true) {
            uint64_t seq = dir_seq_.load(std::memory_order_acquire);
            if (seq & 1) {
                cpu_pause();
                continue;
            }

            int lo = 0;
            int hi = static_cast<int>(bucket_max_.size()) - 1;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                if (bucket_max_[mid].val.load(std::memory_order_relaxed) >= x) hi = mid - 1;
                else lo = mid + 1;
            }
            if (lo >= static_cast<int>(buckets_.size()))
                lo = static_cast<int>(buckets_.size()) - 1;

            if (dir_seq_.load(std::memory_order_acquire) == seq) {
                return lo;
            }
        }
    }

    void update_index(int i) {
        bucket_max_[i].val.store(buckets_[i]->max_key(), std::memory_order_relaxed);
    }

    // Must be called while holding split_lock_
    void split_bucket_global(int i) {
        Bucket* old = buckets_[i];
        
        old->lock.write_lock(); // Must lock old bucket to protect concurrent readers
        int mid = old->count / 2;

        Bucket* right = alloc_bucket();
        int right_count = old->count - mid;
        std::memcpy(right->keys, &old->keys[mid], right_count * sizeof(Key));
        right->count = right_count;
        old->count = mid;

        dir_seq_.fetch_add(1, std::memory_order_release); // Lock Directory
        buckets_.insert(buckets_.begin() + i + 1, right);
        bucket_max_.insert(bucket_max_.begin() + i + 1, AtomicKey<Key>(right->max_key()));
        update_index(i);
        dir_seq_.fetch_add(1, std::memory_order_release); // Unlock Directory
        
        old->lock.write_unlock();
    }

    // Local Flow under locks.
    void flow_locked(int i, int j) {
        Bucket* A = buckets_[i];
        Bucket* B = buckets_[j];

        double dp = A->pressure() - B->pressure();
        if (dp <= EPS_HIGH) return;

        int k = static_cast<int>(C * (dp - EPS_LOW) / 2.0);
        k = std::max(1, std::min(k, static_cast<int>(A->count)));
        k = std::min(k, C - B->count);
        k = std::min(k, A->count - 1);
        if (k <= 0) return;

        if (i < j) {
            std::memmove(&B->keys[k], &B->keys[0], B->count * sizeof(Key));
            std::memcpy(&B->keys[0], &A->keys[A->count - k], k * sizeof(Key));
        } else {
            std::memcpy(&B->keys[B->count], &A->keys[0], k * sizeof(Key));
            std::memmove(&A->keys[0], &A->keys[k], (A->count - k) * sizeof(Key));
        }
        A->count -= k;
        B->count += k;

        update_index(i);
        update_index(j);
    }

    void stabilize(int i) {
        bool active = false;
        for (int step = 0; step < 2; ++step) {
            bool moved = false;

            uint64_t seq = dir_seq_.load(std::memory_order_acquire);
            if (seq & 1) break; // If directory is shifting, skip stabilization to avoid race

            if (i >= static_cast<int>(buckets_.size())) {
                break;
            }

            if (i > 0) {
                int left = i - 1, right = i;
                buckets_[left]->lock.write_lock();
                buckets_[right]->lock.write_lock();

                if (dir_seq_.load(std::memory_order_acquire) == seq) {
                    double dp = buckets_[right]->pressure() - buckets_[left]->pressure();
                    if ((!active && dp > EPS_HIGH) || (active && dp >= EPS_LOW)) {
                        active = true;
                        dir_seq_.fetch_add(1, std::memory_order_release);
                        flow_locked(right, left);
                        dir_seq_.fetch_add(1, std::memory_order_release);
                        moved = true;
                    }
                }
                
                buckets_[right]->lock.write_unlock();
                buckets_[left]->lock.write_unlock();
            }

            if (i + 1 < static_cast<int>(buckets_.size())) {
                int left = i, right = i + 1;
                buckets_[left]->lock.write_lock();
                buckets_[right]->lock.write_lock();

                if (dir_seq_.load(std::memory_order_acquire) == seq) {
                    double dp = buckets_[left]->pressure() - buckets_[right]->pressure();
                    if ((!active && dp > EPS_HIGH) || (active && dp >= EPS_LOW)) {
                        active = true;
                        dir_seq_.fetch_add(1, std::memory_order_release);
                        flow_locked(left, right);
                        dir_seq_.fetch_add(1, std::memory_order_release);
                        moved = true;
                    }
                }

                buckets_[right]->lock.write_unlock();
                buckets_[left]->lock.write_unlock();
            }

            if (!moved) break;
        }
    }

public:
    ConcurrentHydroDS() {
        buckets_.reserve(100000); 
        bucket_max_.reserve(100000);
    }

    ~ConcurrentHydroDS() {
        for (auto* b : buckets_) free_bucket(b);
    }

    void insert(Key x) {
    restart:
        uint64_t seq = dir_seq_.load(std::memory_order_acquire);
        if (seq & 1) { cpu_pause(); goto restart; }
        
        if (buckets_.empty()) {
            std::lock_guard<std::mutex> lock(split_lock_);
            if (buckets_.empty()) {
                Bucket* b = alloc_bucket();
                b->keys[0] = x;
                b->count = 1;
                
                dir_seq_.fetch_add(1, std::memory_order_release);
                buckets_.push_back(b);
                bucket_max_.emplace_back(x);
                dir_seq_.fetch_add(1, std::memory_order_release);
                
                total_size_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            goto restart;
        }

        int i = find_bucket(x);
        Bucket* B = buckets_[i];
        
        B->lock.write_lock();
        
        if (B->needs_split()) {
            B->lock.write_unlock();
            std::lock_guard<std::mutex> lock(split_lock_);
            if (i < static_cast<int>(buckets_.size()) && buckets_[i] == B && B->needs_split()) {
                split_bucket_global(i);
            }
            goto restart;
        }

        if (dir_seq_.load(std::memory_order_acquire) != seq) {
            B->lock.write_unlock();
            goto restart;
        }

        B->insert_sorted(x);
        update_index(i);
        B->lock.write_unlock();
        
        total_size_.fetch_add(1, std::memory_order_relaxed);
        stabilize(i);
    }

    bool search(Key x) const {
    restart:
        uint64_t seq = dir_seq_.load(std::memory_order_acquire);
        if (seq & 1) { cpu_pause(); goto restart; }
        
        if (buckets_.empty()) {
            return false;
        }
        
        int i = find_bucket(x);
        Bucket* B = buckets_[i];
        
        uint64_t v = B->lock.read_lock();
        bool found = B->contains(x);
        
        if (!B->lock.check_version(v) || dir_seq_.load(std::memory_order_acquire) != seq) {
            goto restart;
        }
        
        return found;
    }

    bool erase(Key x) {
    restart:
        uint64_t seq = dir_seq_.load(std::memory_order_acquire);
        if (seq & 1) { cpu_pause(); goto restart; }

        if (buckets_.empty()) {
            return false;
        }

        int i = find_bucket(x);
        Bucket* B = buckets_[i];

        B->lock.write_lock();
        
        int pos = B->lower_bound_pos(x);
        if (pos >= B->count || B->keys[pos] != x) {
            B->lock.write_unlock();
            return false;
        }
        
        if (dir_seq_.load(std::memory_order_acquire) != seq) {
            B->lock.write_unlock();
            goto restart;
        }

        B->remove_at(pos);
        update_index(i);
        
        if (B->is_empty()) {
            B->lock.write_unlock();
            std::lock_guard<std::mutex> lock(split_lock_);
            if (i < static_cast<int>(buckets_.size()) && buckets_[i] == B && B->is_empty()) {
                dir_seq_.fetch_add(1, std::memory_order_release);
                free_bucket(B);
                buckets_.erase(buckets_.begin() + i);
                bucket_max_.erase(bucket_max_.begin() + i);
                dir_seq_.fetch_add(1, std::memory_order_release);
            }
            total_size_.fetch_sub(1, std::memory_order_relaxed);
            return true;
        }

        B->lock.write_unlock();
        total_size_.fetch_sub(1, std::memory_order_relaxed);
        
        if (i > 0 || i + 1 < static_cast<int>(buckets_.size())) {
            stabilize(i);
        }
        return true;
    }

    int64_t range_query(Key lo, Key hi) const {
    restart:
        uint64_t seq = dir_seq_.load(std::memory_order_acquire);
        if (seq & 1) { cpu_pause(); goto restart; }
        
        if (buckets_.empty() || lo > hi) {
            return 0;
        }

        int64_t cnt = 0;
        int i = find_bucket(lo);

        for (; i < static_cast<int>(buckets_.size()); ++i) {
            Bucket* B = buckets_[i];
            
            uint64_t v = B->lock.read_lock();
            if (B->count == 0) {
                if (!B->lock.check_version(v) || dir_seq_.load(std::memory_order_acquire) != seq) {
                    goto restart;
                }
                continue;
            }
            if (B->min_key() > hi) {
                if (!B->lock.check_version(v) || dir_seq_.load(std::memory_order_acquire) != seq) {
                    goto restart;
                }
                break;
            }

            int start = B->lower_bound_pos(lo);
            int local_cnt = 0;
            for (int j = start; j < B->count && B->keys[j] <= hi; ++j) {
                local_cnt++;
            }
            
            if (!B->lock.check_version(v) || dir_seq_.load(std::memory_order_acquire) != seq) {
                goto restart; 
            }
            cnt += local_cnt;
        }
        
        return cnt;
    }

    size_t memory_usage() const {
        size_t bucket_mem = buckets_.size() * sizeof(Bucket);
        size_t dir_mem    = buckets_.capacity() * sizeof(Bucket*);
        size_t idx_mem    = bucket_max_.capacity() * sizeof(AtomicKey<Key>);
        return bucket_mem + dir_mem + idx_mem + sizeof(*this);
    }

    double avg_bucket_fill() const {
        if (buckets_.empty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (const auto* b : buckets_) sum += b->pressure();
        double avg = sum / buckets_.size();
        return avg;
    }

    void print_stats(std::ostream& os = std::cout) const {
        os << "=== ConcurrentHydroDS Stats ===\n"
           << "  Bucket capacity (C): " << C << "\n"
           << "  Total elements:      " << size() << "\n"
           << "  Avg bucket fill:     " << (avg_bucket_fill() * 100.0) << "%\n"
           << "  Memory usage:        " << (memory_usage() / 1024.0 / 1024.0) << " MB\n"
           << "  Bytes/key:           " << (size() == 0 ? 0.0 : static_cast<double>(memory_usage()) / size()) << "\n"
           << "=====================\n";
    }

    size_t size() const {
        return total_size_.load(std::memory_order_relaxed);
    }
};

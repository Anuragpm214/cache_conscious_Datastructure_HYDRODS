#pragma once
// =============================================================================
//  HydroDS — Phase 1: Cache-Friendly, Templated, SIMD-Accelerated
//
//  A pressure-based sorted bucket data structure inspired by fluid dynamics.
//  Elements "flow" between adjacent buckets to maintain balanced load,
//  unlike B+-trees which use discrete split/merge operations.
//
//  Key Optimizations (Phase 1):
//    - Cache-line aligned Bucket structs (no heap-allocated vectors)
//    - memmove/memcpy for intra-bucket operations
//    - Flat bucket_max_ array for cache-friendly bucket lookup
//    - SIMD-accelerated point queries (AVX2, int32/int64)
//    - Tunable bucket capacity via template parameter
//    - Edge-biased insert fast paths (append/prepend)
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// =============================================================================
//  SIMD Helpers (AVX2)
// =============================================================================

namespace hydro_simd {

#ifdef __AVX2__

/// SIMD linear scan for int32 keys on sorted data (with early termination)
inline bool contains_i32(const int32_t* keys, int count, int32_t target) {
    const __m256i t = _mm256_set1_epi32(target);
    int i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys + i));
        __m256i eq = _mm256_cmpeq_epi32(d, t);
        if (_mm256_movemask_epi8(eq)) return true;
        // Early exit: sorted data, if chunk start > target, done
        if (keys[i] > target) return false;
    }
    for (; i < count; ++i) {
        if (keys[i] == target) return true;
        if (keys[i] > target)  return false;
    }
    return false;
}

/// SIMD linear scan for int64 keys on sorted data (with early termination)
inline bool contains_i64(const int64_t* keys, int count, int64_t target) {
    const __m256i t = _mm256_set1_epi64x(target);
    int i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys + i));
        __m256i eq = _mm256_cmpeq_epi64(d, t);
        if (_mm256_movemask_epi8(eq)) return true;
        if (keys[i] > target) return false;
    }
    for (; i < count; ++i) {
        if (keys[i] == target) return true;
        if (keys[i] > target)  return false;
    }
    return false;
}

#endif 
// __AVX2__

} // namespace hydro_simd

// =============================================================================
//  HydroDS Class
// =============================================================================

template <typename Key = int32_t, int BucketCapacity = 256>
class HydroDS {
    static_assert(std::is_arithmetic_v<Key>,
                  "HydroDS Phase 1 supports arithmetic key types only");
    static_assert(BucketCapacity >= 16 && BucketCapacity <= 8192,
                  "BucketCapacity must be in [16, 8192]");

    // -------------------------------------------------------------------------
    //  Constants
    // -------------------------------------------------------------------------
    static constexpr int C        = BucketCapacity;
    double EPS_HIGH = 0.85;   // Flow trigger threshold
    double EPS_LOW  = 0.50;   // Flow target threshold

public:
    void set_thresholds(double high, double low) {
        EPS_HIGH = high;
        EPS_LOW = low;
    }
private:

    // -------------------------------------------------------------------------
    //  Bucket: cache-line aligned, fixed-size key array
    // -------------------------------------------------------------------------
    struct alignas(64) Bucket {
        int32_t count = 0;
        Key keys[C + 1];  // +1 overflow slot for insert-before-split

        // -- Accessors --------------------------------------------------------
        Key max_key() const { return keys[count - 1]; }
        Key min_key() const { return keys[0]; }

        double pressure() const {
            return static_cast<double>(count) / C;
        }
        bool is_empty()    const { return count == 0; }
        bool needs_split() const { return count > C; }

        // -- Binary search (standard, branchful) ------------------------------
        int lower_bound_pos(Key x) const {
            int lo = 0, hi = count;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (keys[mid] < x) lo = mid + 1;
                else               hi = mid;
            }
            return lo;
        }

        // -- Point query ------------------------------------------------------
        bool contains(Key x) const {
#ifdef __AVX2__
            if constexpr (std::is_same_v<Key, int32_t>) {
                return hydro_simd::contains_i32(keys, count, x);
            } else if constexpr (std::is_same_v<Key, int64_t>) {
                return hydro_simd::contains_i64(keys, count, x);
            } else
#endif
            {
                int pos = lower_bound_pos(x);
                return pos < count && keys[pos] == x;
            }
        }

        // -- Sorted insert with edge-biased fast paths -----------------------
        void insert_sorted(Key x) {
            if (count == 0 || x >= keys[count - 1]) {
                // Fast path: append (common for sequential / largest key)
                keys[count++] = x;
                return;
            }
            if (x <= keys[0]) {
                // Fast path: prepend
                std::memmove(&keys[1], &keys[0],
                             static_cast<size_t>(count) * sizeof(Key));
                keys[0] = x;
                ++count;
                return;
            }
            // General: find position and shift right
            int pos = lower_bound_pos(x);
            std::memmove(&keys[pos + 1], &keys[pos],
                         static_cast<size_t>(count - pos) * sizeof(Key));
            keys[pos] = x;
            ++count;
        }

        // -- Remove element at position --------------------------------------
        void remove_at(int pos) {
            if (pos < count - 1) {
                std::memmove(&keys[pos], &keys[pos + 1],
                             static_cast<size_t>(count - pos - 1) * sizeof(Key));
            }
            --count;
        }
    };

    // -------------------------------------------------------------------------
    //  Members
    // -------------------------------------------------------------------------
    std::vector<Bucket*> buckets_;     // Pointers to cache-aligned buckets
    std::vector<Key>     bucket_max_;  // Flat max-key index (cache-friendly lookup)
    size_t               total_size_ = 0;
    size_t               total_buckets_allocated_ = 0;

    // -------------------------------------------------------------------------
    //  Bucket allocation (cache-line aligned)
    // -------------------------------------------------------------------------
    static Bucket* alloc_bucket() {
        // aligned_alloc requires size to be multiple of alignment
        constexpr size_t sz = sizeof(Bucket);
        constexpr size_t align = alignof(Bucket);
        constexpr size_t alloc_sz = ((sz + align - 1) / align) * align;

        void* mem = std::aligned_alloc(align, alloc_sz);
        if (!mem) throw std::bad_alloc();
        return new (mem) Bucket();
    }

    static void free_bucket(Bucket* b) {
        if (b) {
            b->~Bucket();
            std::free(b);
        }
    }

    // -------------------------------------------------------------------------
    //  Bucket lookup: binary search over flat bucket_max_ array
    // -------------------------------------------------------------------------
    int find_bucket(Key x) const {
        int lo = 0;
        int hi = static_cast<int>(bucket_max_.size()) - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (bucket_max_[mid] >= x) hi = mid - 1;
            else                       lo = mid + 1;
        }
        if (lo >= static_cast<int>(buckets_.size()))
            lo = static_cast<int>(buckets_.size()) - 1;
        return lo;
    }

    void update_index(int i) {
        bucket_max_[i] = buckets_[i]->max_key();
    }

    // -------------------------------------------------------------------------
    //  Pressure-based flow: move elements from overfull bucket i to neighbor j
    // -------------------------------------------------------------------------
    void flow(int i, int j) {
        Bucket* A = buckets_[i];
        Bucket* B = buckets_[j];

        double dp = A->pressure() - B->pressure();
        if (dp <= EPS_HIGH) return;

        int k = static_cast<int>(C * (dp - EPS_LOW) / 2.0);
        k = std::max(1, std::min(k, static_cast<int>(A->count)));

        // Safety: don't overflow destination, keep at least 1 in source
        k = std::min(k, C - B->count);
        k = std::min(k, A->count - 1);
        if (k <= 0) return;

        if (i < j) {
            // Move largest k elements of A → front of B
            std::memmove(&B->keys[k], &B->keys[0],
                         static_cast<size_t>(B->count) * sizeof(Key));
            std::memcpy(&B->keys[0], &A->keys[A->count - k],
                        static_cast<size_t>(k) * sizeof(Key));
            A->count -= k;
            B->count += k;
        } else {
            // Move smallest k elements of A → end of B
            std::memcpy(&B->keys[B->count], &A->keys[0],
                        static_cast<size_t>(k) * sizeof(Key));
            std::memmove(&A->keys[0], &A->keys[k],
                         static_cast<size_t>(A->count - k) * sizeof(Key));
            A->count -= k;
            B->count += k;
        }

        update_index(i);
        update_index(j);
    }

    // -------------------------------------------------------------------------
    //  Local stabilization: bounded pressure equalization with neighbors
    // -------------------------------------------------------------------------
    void stabilize(int i) {
        bool active = false;
        for (int step = 0; step < 2; ++step) {
            bool moved = false;

            if (i > 0) {
                double dp = buckets_[i]->pressure() - buckets_[i - 1]->pressure();
                if ((!active && dp > EPS_HIGH) || (active && dp >= EPS_LOW)) {
                    active = true;
                    flow(i, i - 1);
                    moved = true;
                }
            }

            if (i + 1 < static_cast<int>(buckets_.size())) {
                double dp = buckets_[i]->pressure() - buckets_[i + 1]->pressure();
                if ((!active && dp > EPS_HIGH) || (active && dp >= EPS_LOW)) {
                    active = true;
                    flow(i, i + 1);
                    moved = true;
                }
            }

            if (!moved) break;
        }
    }

    // -------------------------------------------------------------------------
    //  Split an overfull bucket at midpoint
    // -------------------------------------------------------------------------
    void split_bucket(int i) {
        Bucket* old = buckets_[i];
        int mid = old->count / 2;

        Bucket* right = alloc_bucket();
        ++total_buckets_allocated_;

        int right_count = old->count - mid;
        std::memcpy(right->keys, &old->keys[mid],
                    static_cast<size_t>(right_count) * sizeof(Key));
        right->count = right_count;
        old->count = mid;

        // Insert new bucket after position i
        buckets_.insert(buckets_.begin() + i + 1, right);
        bucket_max_.insert(bucket_max_.begin() + i + 1, right->max_key());

        update_index(i);
    }

public:
    // =========================================================================
    //  Constructors / Destructor
    // =========================================================================
    HydroDS() {
        buckets_.reserve(16384);
        bucket_max_.reserve(16384);
    }

    ~HydroDS() {
        for (auto* b : buckets_) free_bucket(b);
    }

    // Non-copyable (owns heap-allocated buckets)
    HydroDS(const HydroDS&) = delete;
    HydroDS& operator=(const HydroDS&) = delete;

    // Movable
    HydroDS(HydroDS&& other) noexcept
        : buckets_(std::move(other.buckets_)),
          bucket_max_(std::move(other.bucket_max_)),
          total_size_(other.total_size_),
          total_buckets_allocated_(other.total_buckets_allocated_) {
        other.total_size_ = 0;
        other.total_buckets_allocated_ = 0;
    }

    HydroDS& operator=(HydroDS&& other) noexcept {
        if (this != &other) {
            for (auto* b : buckets_) free_bucket(b);
            buckets_ = std::move(other.buckets_);
            bucket_max_ = std::move(other.bucket_max_);
            total_size_ = other.total_size_;
            total_buckets_allocated_ = other.total_buckets_allocated_;
            other.total_size_ = 0;
            other.total_buckets_allocated_ = 0;
        }
        return *this;
    }

    // =========================================================================
    //  INSERT
    // =========================================================================
    void insert(Key x) {
        if (buckets_.empty()) {
            Bucket* b = alloc_bucket();
            ++total_buckets_allocated_;
            b->keys[0] = x;
            b->count = 1;
            buckets_.push_back(b);
            bucket_max_.push_back(x);
            ++total_size_;
            return;
        }

        int i = find_bucket(x);
        buckets_[i]->insert_sorted(x);
        update_index(i);
        ++total_size_;

        if (buckets_[i]->needs_split()) {
            split_bucket(i);
        }

        stabilize(i);
    }

    // =========================================================================
    //  SEARCH (point query)
    // =========================================================================
    bool search(Key x) const {
        if (buckets_.empty()) return false;
        int i = find_bucket(x);
        return buckets_[i]->contains(x);
    }

    // =========================================================================
    //  ERASE (returns true if element was found and removed)
    // =========================================================================
    bool erase(Key x) {
        if (buckets_.empty()) return false;

        int i = find_bucket(x);
        Bucket* B = buckets_[i];

        int pos = B->lower_bound_pos(x);
        if (pos >= B->count || B->keys[pos] != x) return false;

        B->remove_at(pos);
        --total_size_;

        // Bucket became empty → remove it entirely
        if (B->is_empty()) {
            free_bucket(B);
            buckets_.erase(buckets_.begin() + i);
            bucket_max_.erase(bucket_max_.begin() + i);
            return true;
        }

        update_index(i);

        // Reverse flow: pull elements from overfull neighbors
        if (i > 0) {
            double dp = buckets_[i - 1]->pressure() - buckets_[i]->pressure();
            if (dp > EPS_HIGH) flow(i - 1, i);
        }
        if (i + 1 < static_cast<int>(buckets_.size())) {
            double dp = buckets_[i + 1]->pressure() - buckets_[i]->pressure();
            if (dp > EPS_HIGH) flow(i + 1, i);
        }

        return true;
    }

    // =========================================================================
    //  RANGE QUERY: count elements in [lo, hi]
    // =========================================================================
    int64_t range_query(Key lo, Key hi) const {
        if (buckets_.empty() || lo > hi) return 0;

        int64_t cnt = 0;
        int i = find_bucket(lo);

        for (; i < static_cast<int>(buckets_.size()); ++i) {
            const Bucket* B = buckets_[i];
            if (B->count == 0) continue;
            if (B->min_key() > hi) break;

            // Binary search for start within this bucket
            int start = B->lower_bound_pos(lo);
            for (int j = start; j < B->count && B->keys[j] <= hi; ++j) {
                ++cnt;
            }
        }
        return cnt;
    }

    // =========================================================================
    //  RANGE COLLECT: return elements in [lo, hi]
    // =========================================================================
    std::vector<Key> range_collect(Key lo, Key hi) const {
        std::vector<Key> result;
        if (buckets_.empty() || lo > hi) return result;

        int i = find_bucket(lo);
        for (; i < static_cast<int>(buckets_.size()); ++i) {
            const Bucket* B = buckets_[i];
            if (B->count == 0) continue;
            if (B->min_key() > hi) break;

            int start = B->lower_bound_pos(lo);
            for (int j = start; j < B->count && B->keys[j] <= hi; ++j) {
                result.push_back(B->keys[j]);
            }
        }
        return result;
    }

    // =========================================================================
    //  Statistics & Diagnostics
    // =========================================================================
    size_t size()         const { return total_size_; }
    bool   empty()        const { return total_size_ == 0; }
    size_t bucket_count() const { return buckets_.size(); }

    /// Total memory used by the data structure (bytes)
    size_t memory_usage() const {
        size_t bucket_mem = buckets_.size() * sizeof(Bucket);
        size_t dir_mem    = buckets_.capacity() * sizeof(Bucket*);
        size_t idx_mem    = bucket_max_.capacity() * sizeof(Key);
        return bucket_mem + dir_mem + idx_mem + sizeof(*this);
    }

    /// Bytes per key (amortized)
    double bytes_per_key() const {
        if (total_size_ == 0) return 0.0;
        return static_cast<double>(memory_usage()) / total_size_;
    }

    /// Average bucket fill ratio (0.0 to 1.0)
    double avg_bucket_fill() const {
        if (buckets_.empty()) return 0.0;
        double sum = 0.0;
        for (const auto* b : buckets_) sum += b->pressure();
        return sum / buckets_.size();
    }

    /// Min/Max bucket fill
    std::pair<double, double> bucket_fill_range() const {
        if (buckets_.empty()) return {0.0, 0.0};
        double lo = 1.0, hi = 0.0;
        for (const auto* b : buckets_) {
            double p = b->pressure();
            lo = std::min(lo, p);
            hi = std::max(hi, p);
        }
        return {lo, hi};
    }

    /// Print diagnostics
    void print_stats(std::ostream& os = std::cout) const {
        auto [fill_lo, fill_hi] = bucket_fill_range();
        os << "=== HydroDS Stats ===\n"
           << "  Bucket capacity (C): " << C << "\n"
           << "  Total elements:      " << total_size_ << "\n"
           << "  Bucket count:        " << buckets_.size() << "\n"
           << "  Avg bucket fill:     " << (avg_bucket_fill() * 100.0) << "%\n"
           << "  Fill range:          [" << (fill_lo * 100.0) << "%, "
                                         << (fill_hi * 100.0) << "%]\n"
           << "  Memory usage:        " << (memory_usage() / 1024.0 / 1024.0)
                                         << " MB\n"
           << "  Bytes/key:           " << bytes_per_key() << "\n"
           << "  Key size:            " << sizeof(Key) << " bytes\n"
           << "=====================\n";
    }

    // =========================================================================
    //  Internal integrity check (for testing)
    // =========================================================================
    bool verify_integrity() const {
        if (buckets_.size() != bucket_max_.size()) return false;

        size_t counted = 0;
        for (size_t bi = 0; bi < buckets_.size(); ++bi) {
            const Bucket* B = buckets_[bi];
            if (B->count <= 0) return false;  // No empty buckets allowed

            // Keys must be sorted within bucket
            for (int j = 1; j < B->count; ++j) {
                if (B->keys[j] < B->keys[j - 1]) return false;
            }

            // bucket_max_ must match
            if (bucket_max_[bi] != B->max_key()) return false;

            // Buckets must be in order: max of bucket i < min of bucket i+1
            // (unless there are duplicates spanning buckets)
            if (bi + 1 < buckets_.size()) {
                if (B->max_key() > buckets_[bi + 1]->min_key()) return false;
            }

            counted += B->count;
        }

        return counted == total_size_;
    }
};

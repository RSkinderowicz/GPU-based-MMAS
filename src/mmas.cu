/**
  @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
*/
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cfloat>
#include <ctime>
#include <random>
#include <sys/stat.h> // stat
#include <errno.h>    // errno, ENOENT, EEXIST

#include <cuda.h>
#include <curand_kernel.h>

#include "tsp.h"
#include "mmas.h"
#include "json.hpp"
#include "utils.h"

#define WARP_SIZE 32

// For working with all threads in a warp:
#define FULL_MASK 0xffffffff


// Utility for time measurements
struct Timer {
    using Clock = std::chrono::high_resolution_clock;

    Timer() : start_time_(Clock::now()) {}

    // Time since the constructor was called
    double get_elapsed_seconds() const noexcept {
        return get_elapsed_nanoseconds() * 1e-9;
    }

    int64_t get_elapsed_nanoseconds() const noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>
            (Clock::now() - start_time_).count();
    }

    Clock::time_point start_time_;
};


/**
 * Returns a number in range [min, max] drawn randomly with uniform
 * probability.
 */
__device__
static inline uint32_t get_random_uint32(curandState_t &state,
                                         uint32_t min, uint32_t max) {
    const auto range = max - min;
    uint32_t x = curand(&state);
    uint64_t m = uint64_t(x) * uint64_t(range);
    uint32_t l = uint32_t(m);
    if (l < range) {
        uint32_t t = -range;
        if (t >= range) {
            t -= range;
            if (t >= range) 
                t %= range;
        }
        while (l < t) {
            x = curand(&state);
            m = uint64_t(x) * uint64_t(range);
            l = uint32_t(m);
        }
    }
    return m >> 32;
}


__device__
static inline float get_random_float(curandState_t &s){
    return curand_uniform(&s);
}


using rand_state_t = curandState_t;


__device__ int32_t get_warp_count() { return blockDim.x / WARP_SIZE; }

__device__ int32_t get_warp_id() { return threadIdx.x / WARP_SIZE; }

__device__ int32_t get_block_count() { return gridDim.x; }

__device__ int32_t get_global_thread_id() {
     return blockIdx.x * blockDim.x + threadIdx.x; 
}

/*
  Broadcast value from leader thread to other threads in a warp
 */ 
template<class T>
__device__ __forceinline__
T warp_bcast(T value, int leader) { 
    return __shfl_sync(FULL_MASK, value, leader); 
}

 
/**
Returns an inclusive scan.
*/
template<typename T, int warp_size>
__device__ __forceinline__
T warp_scan(T value, uint32_t mask = FULL_MASK) {
    T my_sum = value;
    for (int i = 1; i < warp_size; i *= 2) {
        const T other = __shfl_up_sync(mask, my_sum, i);
        if (threadIdx.x % warp_size >= i) my_sum += other;
    }
    return my_sum;
}


template<typename T>
__device__ __forceinline__
T warp_scan(T value) {
    return warp_scan<T, WARP_SIZE>(value, FULL_MASK);
}


/**
Warp-level reduction. Only the first thread, i.e. with threadIdx.x = 0
get the final result.
*/
template<typename T, int warp_size>
__device__ __forceinline__
T warp_reduce(T value) {
    T my_sum = value;
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        my_sum += __shfl_down_sync(FULL_MASK, my_sum, offset);
    }
    return my_sum;
}


template<typename arg_t, typename value_t, int warp_size>
__device__
arg_t warp_reduce_arg_max(arg_t arg, value_t val,
                          uint32_t active_threads_mask = FULL_MASK) {
    value_t my_max = val;
    // Find max value using warp level shuffling
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        const auto other_max = __shfl_down_sync(active_threads_mask, my_max, offset);
        const auto other_arg = __shfl_down_sync(active_threads_mask, arg, offset);
        arg = (my_max < other_max) ? other_arg : arg; 
        my_max = max(my_max, other_max);
    }
    return arg;
}

/**
Warp level summation - each thread gets the result.
*/
__inline__ __device__
float warp_all_reduce_sum(float val) {
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        val+= __shfl_xor_sync(FULL_MASK, val, i);
    }
    return val;
}


/**
Warp-level reduction for maximum value. Each thread gets the max.
*/
template<typename T>
__device__
float warp_all_reduce_max(T val) {
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        val = max(val, __shfl_xor_sync(FULL_MASK, val, i));
    }
    return val;
}


/**
Returns a position of the first (starting from 1 up to WARP_SIZE) thread for
which thread_vote is true or 0 otherwise, i.e. if all threads' votes are false.
Only threads for which the corresponding bit is set in active_threads_mask
are voting.
*/
__device__
uint32_t warp_vote(bool thread_vote, uint32_t active_threads_mask=FULL_MASK) {
    return __ffs(__ballot_sync(active_threads_mask, thread_vote));
}


__device__
uint32_t warp_vote_last(bool thread_vote, uint32_t active_threads_mask=FULL_MASK) {
    return WARP_SIZE - __clz( __ballot_sync(active_threads_mask, thread_vote) );
}


void cuda_assert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA assertion failed: "
                  << cudaGetErrorString(code)
                  << " at: " << file
                  << ":"
                  << line << std::endl;

        exit(code);
    }
}


/**
Broadcasts a value from thread with id equal to src_thread_id to all threads
in a block.
*/
template<typename T>
__device__
T block_bcast(T value, int32_t src_thread_id) {
    __shared__ T result;

    if (threadIdx.x == src_thread_id) {
        result = value;
    }
    __syncthreads();
    return result;
}


/**
Only the thread with threadIdx.x == 0 is guaranteed to have the final result.
*/
__device__
uint32_t block_arg_max(uint32_t arg, float value, char *shared_buffer) {
    const auto warp_count = get_warp_count();

    if (warp_count == 1) {
        arg = warp_reduce_arg_max<uint32_t, float, WARP_SIZE>(arg, value);
        return warp_bcast(arg, 0);
    }
    // Current impl. is limited to block with no more than WARP_SIZE warps
    assert(warp_count <= WARP_SIZE);

    const auto warp_id = threadIdx.x / WARP_SIZE;
    float *chosen_values = (float *)shared_buffer;
    uint32_t *chosen_args = (uint32_t *)(shared_buffer
                                       + sizeof(float) * warp_count);

    float warp_max_key = warp_all_reduce_max(value);

    if (value == warp_max_key) {
        chosen_args[warp_id] = arg;
        chosen_values[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {  // Scan the partial (warps') sums using only the first warp
        const auto lane = threadIdx.x % WARP_SIZE;
        value = (threadIdx.x < warp_count) ? chosen_values[lane] : warp_max_key;
        arg = (threadIdx.x < warp_count) ? chosen_args[lane] : arg;
        arg = warp_reduce_arg_max<uint32_t, float, WARP_SIZE>(arg, value);
    }
    return block_bcast(arg, /*src. thread */0);
}


__device__
float block_sum(float value, char *shared_buffer) {
    const auto warp_count = blockDim.x / WARP_SIZE;

    assert(warp_count <= WARP_SIZE);

    const auto warp_id = threadIdx.x / WARP_SIZE;
    float *warp_sums = (float *)shared_buffer;

    warp_sums[warp_id] = warp_all_reduce_sum(value);

    __syncthreads();

    __shared__ float block_sum;

    if (warp_id == 0) {
        value = (threadIdx.x < warp_count) ? warp_sums[threadIdx.x] : 0;
        block_sum = warp_all_reduce_sum(value);
    }
    __syncthreads();
    return block_sum;
}


/**
Uses atomic instructions to select argument corresponding to maximum key.
*/
__device__
uint32_t block_arg_max_atomics(uint32_t arg, float key) {
    // For a single warp we do not need to use (slower) atomics
    if (get_warp_count() == 1) {
        const auto result = warp_reduce_arg_max<uint32_t, float, WARP_SIZE>(arg, key);
        return warp_bcast<uint32_t>(result, 0);
    }

    const auto warp_max_key = warp_all_reduce_max(key);

    __shared__ float block_max_key;

    if (get_warp_id() == 0) {
        block_max_key = warp_max_key;
    }

    __syncthreads();

    const auto lane_id = threadIdx.x % WARP_SIZE;
    const auto is_first_thread_in_warp = lane_id == 0;

    if (is_first_thread_in_warp) {
        int* address_as_i = reinterpret_cast<int*>( &block_max_key );
        int old = *address_as_i;
        int assumed;
        do {
            assumed = old;
            if (__int_as_float(assumed) < warp_max_key) {
                old = ::atomicCAS(address_as_i,
                                assumed,
                                __float_as_int(warp_max_key));
            } else {
                break ;
            }
        } while (assumed != old);
    }

    __syncthreads();

    __shared__ uint32_t chosen_arg;

    if (block_max_key == key) {
        chosen_arg = arg;
    }

    __syncthreads();

    return chosen_arg;
}


__device__
float block_scan(float thread_value) {
    assert( get_warp_count() <= WARP_SIZE );

    const auto lane = threadIdx.x % WARP_SIZE;
    const auto warp_prefix_sum = warp_scan(thread_value);

    __shared__ float sums[WARP_SIZE];

    if (lane + 1 == WARP_SIZE) {
        sums[0] = 0;
        sums[get_warp_id() + 1] = warp_prefix_sum;
    }

    __syncthreads();

    if (get_warp_id() == 0) {
        const auto my_sum = (lane < get_warp_count()) ? sums[lane] : 0;
        const auto prefix_sum = warp_scan(my_sum);

        sums[lane] = prefix_sum;
    }

    __syncthreads();

    return sums[get_warp_id()] + warp_prefix_sum;
}


/**
Returns a position (from 1 to blockDim.x) of the first thread for which
thread_vote is true. If no such thread exists 0 is returned.
works for blocks of threads smaller than 2^30.
*/
__device__
int32_t block_vote(bool thread_vote) {
    assert(blockDim.x < (1 << 30));

    const auto warp_winner_pos = warp_vote(thread_vote);

    __shared__ int32_t block_winner_pos;

    if (get_warp_id() == 0) {
        block_winner_pos = 1 << 30;
    }

    __syncthreads();

    if (warp_winner_pos > 0) {
        const auto pos_in_block = warp_winner_pos + get_warp_id() * WARP_SIZE;
        atomicMin(&block_winner_pos, pos_in_block);
    }

    __syncthreads();

    return block_winner_pos % (1 << 30);
}


/**
Returns a position (from 1 to blockDim.x) of the last thread for which
thread_vote is true. If no such thread exists 0 is returned.
Works for blocks of any size.
*/
__device__
int32_t block_vote_last(bool thread_vote) {
    const auto warp_winner_pos = warp_vote_last(thread_vote);

    __shared__ int32_t block_winner_pos;

    if (get_warp_id() == 0) {
        block_winner_pos = 0;
    }

    __syncthreads();

    if (warp_winner_pos > 0) {
        const auto pos_in_block = warp_winner_pos + get_warp_id() * WARP_SIZE;
        atomicMax(&block_winner_pos, pos_in_block);
    }

    __syncthreads();

    return block_winner_pos;
}


/**
 * Works like the CUDA's built in atomicMax but for floats.
 */
__device__
float atomicMax(float *address, float value) {
    int* address_as_i = reinterpret_cast<int*>(address);
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) < value) {
            old = ::atomicCAS(address_as_i,
                            assumed,
                            __float_as_int(value));
        } else {
            break ;
        }
    } while (assumed != old);
    return __int_as_float(old);
}


#define CUDA_CHECK(ret_code) { cuda_assert((ret_code), __FILE__, __LINE__); }


/**
Auxiliary class simplifying copying of arrays between the host and device (GPU)
*/
template<typename T>
class device_vector {
public:

    /**
    Creates a device-side, i.e. in the GPU memory, vector with size 
    elements of type T 
    */
    device_vector(size_t size) :
        size_(size) {

        const auto bytes = size * sizeof(T);
        // std::cout << "Allocating " << bytes << " bytes on the GPU" << std::endl;
        CUDA_CHECK( cudaMalloc((void **)&d_ptr_, bytes) );
    }

    /**
    Creates a device-side vector with size elements of type T.
    Elements of the vector are initialized to the given value.
    */
    device_vector(size_t size, const T &initial_value) :
        device_vector(size) {

        // First create a temporary vector on the host side and fill it with
        // initial_value
        std::vector<T> host_vec(size, initial_value);
        // Now copy the vector to the device
        const auto bytes = size * sizeof(T);
        CUDA_CHECK( cudaMemcpy(d_ptr_, host_vec.data(),
                               bytes,
                               cudaMemcpyHostToDevice) );
    }

    /**
     * Creates a vector in the GPU memory with the contents copied from src.
     */
    device_vector(const std::vector<T> &src) :
        device_vector(src.size()) {

        const auto bytes = size_ * sizeof(T);
        CUDA_CHECK( cudaMemcpy(d_ptr_, src.data(),
                               bytes,
                               cudaMemcpyHostToDevice) );
    }

    virtual ~device_vector() {
        if (d_ptr_) {
            CUDA_CHECK( cudaFree(d_ptr_) );
            d_ptr_ = nullptr;
        }
    }

    T* data() noexcept { return d_ptr_; }

    operator T*() noexcept { return d_ptr_; }

    size_t size() const noexcept { return size_; }

    void copy_to_host(std::vector<T> &dest) const {
        dest.resize(size_);
        CUDA_CHECK( cudaMemcpy(dest.data(),
                               d_ptr_,
                               size_ * sizeof(T),
                               cudaMemcpyDeviceToHost) );
    }

    /**
    Returns a host-side copy of the vector, i.e. copies data from the GPU memory
    */
    std::vector<T> as_host_vector() const {
        std::vector<T> dest(size_);
        copy_to_host(dest);
        return dest;
    }

    void copy_to_device(const std::vector<T> &host_vec) {
        assert(size_ == host_vec.size());

        const auto bytes = size_ * sizeof(T);
        CUDA_CHECK( cudaMemcpy(d_ptr_, host_vec.data(), bytes,
                               cudaMemcpyHostToDevice) );
    }

private:
    size_t size_ = 0;
    T *d_ptr_ = nullptr;
}; 


// Utility for time measurements using CUDA utilities
struct GPUTimer {

    GPUTimer() {
        CUDA_CHECK( cudaEventCreate(&start_) );
        CUDA_CHECK( cudaEventCreate(&stop_) );
    }

    ~GPUTimer() {
        CUDA_CHECK( cudaEventDestroy(start_) );
        CUDA_CHECK( cudaEventDestroy(stop_) );
    }

    void start() {
        CUDA_CHECK( cudaEventRecord(start_) );
    }

    /**
    * Returns time since the last start() call
    */
    int64_t get_elapsed_nanoseconds_since_start() const noexcept {
        CUDA_CHECK( cudaEventRecord(stop_) );
        CUDA_CHECK( cudaEventSynchronize(stop_) );

        float elapsed_ms;
        CUDA_CHECK( cudaEventElapsedTime(&elapsed_ms, start_, stop_) );
        return int64_t(elapsed_ms * 1e6);
    }

    void accumulate_elapsed() {
        total_elapsed_ += get_elapsed_nanoseconds_since_start();
        ++accumulate_calls_;
    }

    double get_total_elapsed_seconds() const noexcept {
        return total_elapsed_ * 1e-9;
    }

    double get_mean_elapsed_seconds() const noexcept {
        if (accumulate_calls_ > 0) {
            return get_total_elapsed_seconds() / accumulate_calls_;
        }
        return 0;
    }

    cudaEvent_t start_;
    cudaEvent_t stop_;
    int64_t total_elapsed_ = 0;
    int64_t accumulate_calls_ = 0;
};


struct Ant {
    std::vector<uint32_t> visited_; // A list of visited nodes, i.e. a route
    std::vector<uint8_t> is_visited_;
    double cost_ = std::numeric_limits<double>::max();

    void initialize(uint32_t dimension) {
        visited_.clear();
        visited_.reserve(dimension);
        is_visited_.clear();
        is_visited_.resize(dimension, false);
    }

    void visit(uint32_t node) {
        assert(!is_visited(node));

        visited_.push_back(node);
        is_visited_[node] = true;
    }

    bool is_visited(uint32_t node) const {
        assert(node < is_visited_.size());

        return is_visited_[node];
    }
};


/**
 * This are based on the article mentioned.
 */
struct MMASParameters {
    double rho_ = 0.99;
    uint32_t ants_count_ = 10;
    double beta_ = 2;
    bool use_cand_lists_ = true;
    uint32_t cand_list_size_ = 32;
    double p_best_ = 0.01; // Prob. that the constructed sol. will contain only
                           // the edges with the highest pheromone values

    __host__ __device__
    double get_evaporation_rate() const { return 1 - rho_; }
};


struct TrailLimits {
    float min_ = 0;
    float max_ = 0;
};


/**
 * This is based on Eq. 11 from the original MAX-MIN paper:
 *
 * Stützle, Thomas, and Holger H. Hoos. "MAX–MIN ant system." Future generation
 * computer systems 16.8 (2000): 889-914.
 */
__host__ __device__
TrailLimits calc_trail_limits(uint32_t instance_dimension,
                                   float solution_cost,
                                   float evaporation_rate = 0.01f,
                                   float p_best = 0.05f) {
    const auto tau_max = 1 / (solution_cost * evaporation_rate);
    const auto avg = instance_dimension / 2.0f;
    const auto p = std::pow(p_best, 1.0f / instance_dimension);
    const auto r = tau_max * (1 - p) / ((avg - 1) * p);
    const auto tau_min = r < tau_max ? r : tau_max;
    return TrailLimits{ tau_min, tau_max };
}


/**
Gathers problem instance-related data to lower number of parameters 
passed to kernels.

Warning: All pointers should point to device-allocated memory.
*/
struct InstanceContext {
    uint32_t dimension_ = 0;
    float* coordinates_ = nullptr;  // Location (x, y) of each node / city of the TSP
    float* distance_matrix_ = nullptr;
    float* heuristic_matrix_ = nullptr;
    float heuristic_weight_ = 2;  // importance of the heuristic information
    uint32_t cand_list_size_ = 0;
    uint32_t* cand_lists_ = nullptr;

    __device__ 
    float get_distance(uint32_t from,
                       uint32_t to) const {

        assert(from < dimension_ && to < dimension_);
        assert(distance_matrix_ != nullptr);

        return distance_matrix_[from * dimension_ + to];
    }
};


/**
Gathers MMAS-related variables to lower the number of parameters
passed to kernels.

Warning: All pointers should point to device-allocated memory.
*/
struct MMASRunContext {
    uint32_t ants_count_ = 0;

    float *pheromone_matrix_ = nullptr;
    float *product_cache_ = nullptr;
    float *cand_lists_product_cache_ = nullptr;

    uint32_t *ant_routes_ = nullptr;
    float *ant_route_costs_ = nullptr;

    uint32_t *iter_best_route_ = nullptr;
    float *iter_best_cost_ = nullptr;

    uint32_t *global_best_route_ = nullptr;
    float *global_best_cost_ = nullptr;

    // If pheromone resetting is used then this stores the best solution found
    // since the reset
    uint32_t *reset_best_route_ = nullptr;
    float *reset_best_cost_ = nullptr;

    constexpr static uint32_t global_best_log_capacity = 5000;
    // Stores (value, iter.) pair for every global best solution found
    // The size is limited to 1000 entries
    std::pair<float, uint32_t> *global_best_log_ = nullptr;
    uint32_t *global_best_log_length_ = nullptr;  // Just the length
};

/**
Updates nn_product_cache by loading product of heuristic and pheromone values.
One block is responsible for one node, and a thread in the block is responsible
for one neigboring node.
*/
__global__
void update_cand_lists_pheromone_heuristic_product_cache(
        InstanceContext instance,
        MMASRunContext ctx,
        bool calc_reciprocal) {

    assert(instance.dimension_ == get_block_count());

    const auto src_node = blockIdx.x;
    const auto cand_list_size = instance.cand_list_size_;

    const auto offset = src_node * instance.dimension_;
    auto *heuristic = instance.heuristic_matrix_ + offset;
    auto *pheromone = ctx.pheromone_matrix_ + offset;

    const auto cand_list_offset = src_node * cand_list_size;
    auto *cand_list = instance.cand_lists_ + cand_list_offset;
    auto *product_cache = ctx.cand_lists_product_cache_ + cand_list_offset;

    for (uint32_t i = threadIdx.x; i < cand_list_size; i += blockDim.x) {
        const auto node = cand_list[i];
        const auto product = heuristic[node] * pheromone[node];
        product_cache[i] = (calc_reciprocal && product > 0)
                         ? 1.0f / product
                         : product;
    }
}


/**
Updates product_cache by storing the product of heuristic and pheromone values.
One block is responsible for one node, and a thread in the block is responsible
for one neigboring node.
*/
__global__
void update_pheromone_heuristic_product_cache(
        InstanceContext instance,
        MMASRunContext ctx,
        bool calc_reciprocal) {

    assert(instance.dimension_ == get_block_count());

    const auto node = blockIdx.x;
    const auto offset = node * instance.dimension_;
    const auto *heuristic = instance.heuristic_matrix_ + offset;
    const auto *pheromone = ctx.pheromone_matrix_ + offset;
    auto *product_cache = ctx.product_cache_ + offset;

    for (uint32_t endpoint = threadIdx.x;
         endpoint < instance.dimension_;
         endpoint += blockDim.x) {

        const auto product = heuristic[endpoint] * pheromone[endpoint];
        const auto result = (calc_reciprocal && product > 0)
                          ? (1 / product)
                          : product;
        product_cache[endpoint] = result;
    }
}


template<typename T>
__host__ __device__
T round_to_multiple_of(const T n, T x) {
    return (x / n + (x % n != 0 ? 1 : 0)) * n;
}


struct BitmaskTabu {
    const uint32_t dimension_;
    uint32_t visited_count_;

    uint32_t *visited_mask_;  // This is in the shared memory
    uint32_t *out_route_;     // This should point to the global memory
    // These are used to store a chunk of visited nodes in the threads' registers
    // for a delayed bulk commit to the main (global) memory, i.e. out_route_
    uint32_t out_node_;
    uint32_t out_count_;

    /**
     * dimension refers to a problem's dimension
     */
    __device__
    BitmaskTabu(uint32_t dimension, char *shared_buffer, uint32_t *out_route)
        : dimension_(dimension),
          visited_mask_(reinterpret_cast<uint32_t *>(shared_buffer)),
          out_route_(out_route),
          visited_count_(0)
    {}

    __host__ __device__
    static uint32_t get_required_shared_memory_size_in_bytes(uint32_t dimension) {
        return round_to_multiple_of(4u, get_mask_size_in_bytes(dimension));
    }

    __host__ __device__
    static uint32_t get_mask_size_in_bytes(uint32_t dimension) {
        const auto bits_per_word = sizeof(uint32_t) * 8;
        return (dimension / bits_per_word
                + (dimension % bits_per_word != 0 ? 1 : 0)) * sizeof(uint32_t);
    }

    __device__
    void parallel_init() {
        const auto words =
          BitmaskTabu::get_mask_size_in_bytes(dimension_) / sizeof(uint32_t);
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) {
            visited_mask_[i] = 0;
        }
        visited_count_ = 0;
        out_count_ = 0;
        __syncthreads();
    }

    __device__
    void parallel_finalize() {
        if (threadIdx.x < out_count_) {
            out_route_[visited_count_ - out_count_ + threadIdx.x] = out_node_;
        }
        __syncthreads();

    }

    __device__
    void add_visited(uint32_t node) {
        if (threadIdx.x == 0) {
            assert(visited_count_ < dimension_);
            visited_mask_[node / 32] |= 1 << (node % 32);
        }
        ++visited_count_;

        if (out_count_ == threadIdx.x) {
            out_node_ = node;  // Save this node for the later commit to out_route_
        }

        ++out_count_;

        // Is this time to commit gathered nodes to out_route_?
        if (out_count_ == blockDim.x) {  
            out_route_[visited_count_ - blockDim.x + threadIdx.x] = out_node_;
            out_count_ = 0;
        }
        __syncthreads();
    }

    __device__
    bool is_visited(uint32_t node) const {
        assert(node < dimension_);
        return (visited_mask_[node / 32]) & (1 << (node % 32));
    }

    __device__
    bool is_available(uint32_t node) const { return !is_visited(node); }

    __device__
    uint32_t get_length() const { return dimension_; }

    /**
     * Returns a candidate node corresponding to the given index or
     * get_length() if the node at index was already visited.
     */
    __device__
    uint32_t get_candidate(uint32_t index) const {
        assert(index < dimension_);
        return index;
    }

    __device__
    bool is_candidate_unvisited(uint32_t cand) const { 
        return is_available(cand);
    }
};


struct CompressedListTabu {
    const uint32_t dimension_;
    uint32_t length_;
    uint16_t *unvisited_;
    uint16_t *indices_;
    uint32_t *out_route_;

    /**
     * dimension refers to a problem's dimension
     */
    __device__
    CompressedListTabu(uint32_t dimension, char *shared_buffer, uint32_t *out_route)
        : dimension_(dimension),
          length_(dimension),
          unvisited_(reinterpret_cast<uint16_t *>(shared_buffer)),
          indices_(reinterpret_cast<uint16_t *>(shared_buffer
                                                       + dimension * sizeof(uint16_t))),
          out_route_(out_route)
    {}
    
    __host__ __device__
    static uint32_t get_required_shared_memory_size_in_bytes(uint32_t dimension) {
        return 2 * dimension * sizeof(uint16_t);
    }

    __device__
    void parallel_init() {
        for (uint32_t i = threadIdx.x; i < dimension_; i += blockDim.x) {
            unvisited_[i] = i;
            indices_[i] = i;
        }
        length_ = dimension_;
        __syncthreads();
    }

    /**
     * This saves the visited nodes to the array pointed by out_route_ member.
     */
    __device__
    void parallel_finalize() {
        for (uint32_t i = threadIdx.x; i < dimension_; i += blockDim.x) {
            out_route_[i] = unvisited_[dimension_ - 1 - i];
        }
        __syncthreads();
    }

    /**
    This can be called by all threads but only the first one modifies the
    tabu
    */
    __device__
    void add_visited(uint32_t node) {
        if (threadIdx.x == 0) {
            assert(length_ > 0);
            assert(is_available(node));

            const auto tail_node = unvisited_[length_ - 1];
            const auto node_index = indices_[node];
            unvisited_[node_index] = tail_node;
            unvisited_[length_ - 1] = node;  // Swapped with tail_node
            indices_[tail_node] = node_index;
            indices_[node] = length_ - 1;  // This marks node as visited
        }
        --length_;
        __syncthreads();
    }

    __device__
    bool is_visited(uint32_t node) const {
        return indices_[node] >= get_length();
    }

    __device__
    bool is_available(uint32_t node) const { return !is_visited(node); }

    __device__
    uint32_t get_length() const { return length_; }

    /**
     * Returns a candidate node corresponding to the given index.
     */
    __device__
    uint32_t get_candidate(uint32_t index) const {
        assert(index < get_length());
        return unvisited_[index];
    }

    __device__
    bool is_candidate_unvisited(uint32_t) const { return true; }
};


struct CompactTabu {
    const uint32_t dimension_;
    uint32_t length_;
    
    uint16_t *unvisited_;  // This points to the shared memory
    uint32_t *out_route_;  // This points to the global memory

    // These are used to store a chunk of visited nodes in the threads' registers
    // for a delayed bulk commit to the main (global) memory, i.e. out_route_
    uint32_t out_node_;
    uint32_t out_count_;

    /**
     * dimension refers to a problem's dimension
     */
    __device__
    CompactTabu(uint32_t dimension, char *shared_buffer, uint32_t *out_route)
        : dimension_(dimension),
          length_(dimension),
          unvisited_(reinterpret_cast<uint16_t *>(shared_buffer)),
          out_route_(out_route),
          out_count_(0)
    {}
    
    __host__ __device__
    static uint32_t get_required_shared_memory_size_in_bytes(uint32_t dimension) {
        return round_to_multiple_of(4ul, dimension * sizeof(uint16_t));
    }

    __device__
    void parallel_init() {
        for (uint32_t i = threadIdx.x; i < dimension_; i += blockDim.x) {
            unvisited_[i] = i;
        }
        length_ = dimension_;
        out_count_ = 0;
        __syncthreads();
    }

    /**
     * This saves the visited nodes to the array pointed by out_route_ member.
     */
    __device__
    void parallel_finalize() {
        if (threadIdx.x < out_count_) {
            out_route_[get_visited_count() - out_count_ + threadIdx.x] = out_node_;
        }
        __syncthreads();
    }

    /**
    This can be called by all threads but only the first one modifies the
    tabu
    */
    __device__
    void add_visited(uint32_t node) {
        if (threadIdx.x == 0) {
            assert(is_available(node));
            assert(length_ > 0);

            const auto tail_index = get_length() - 1;
            const auto tail_node = unvisited_[tail_index];
            const auto node_index = unvisited_[node];

            assert(unvisited_[node_index] == node);
            assert(unvisited_[unvisited_[tail_index]] == tail_index);

            if (node_index < tail_index) {
                unvisited_[node] = dimension_;  // We use dimension_ to mark an empty entry
                unvisited_[node_index] = tail_node;
                unvisited_[tail_index] = dimension_;
                unvisited_[tail_node] = node_index;

                assert(unvisited_[unvisited_[tail_node]] == tail_node);
            } else {  // We are removing the tail node
                assert(node_index == tail_index);
                assert(unvisited_[unvisited_[tail_node]] == tail_node);

                unvisited_[tail_node] = dimension_;  // We use dimension_ to mark an empty entry 
                unvisited_[tail_index] = dimension_;
            }
        }
        --length_;

        if (out_count_ == threadIdx.x) {
            out_node_ = node;
        }

        ++out_count_;

        if (out_count_ == blockDim.x) {
            out_route_[get_visited_count() - blockDim.x + threadIdx.x] = out_node_;
            out_count_ = 0;
        }
        __syncthreads();
    }

    __device__
    bool is_visited(uint32_t node) const {
        assert(node < dimension_);
        return unvisited_[node] > node;
    }

    __device__
    bool is_available(uint32_t node) const { return !is_visited(node); }

    __device__
    uint32_t get_length() const { return length_; }

    __device__
    uint32_t get_visited_count() const { return dimension_ - length_; }

    /**
     * Returns a candidate node corresponding to the given index.
     */
    __device__
    uint32_t get_candidate(uint32_t index) const {
        assert(index < get_length());
        return unvisited_[index];
    }

    __device__
    bool is_candidate_unvisited(uint32_t) const { return true; }
};


template<typename Tabu>
__device__
uint32_t warp_roulette_choice_from_cand_list(
        InstanceContext instance,
        rand_state_t &rng,
        uint32_t current_node,
        float *cand_lists_product_cache,
        Tabu &tabu) {

    // One thread per candidate element
    assert(blockDim.x == instance.cand_list_size_);
    assert(get_warp_count() == 1);

    const auto offset = instance.cand_list_size_ * current_node;
    const auto cand_list = instance.cand_lists_ + offset;
    const auto cand_index = threadIdx.x;
    const auto cand_node = cand_list[cand_index];
    const bool is_visited = tabu.is_visited(cand_node);
    const auto product_cache = cand_lists_product_cache + offset;
    const auto product = is_visited ? 0 : product_cache[cand_index];

    uint32_t chosen_node = instance.dimension_;

    const auto prefix_sum = warp_scan(product);  // Inclusive
    const auto unvisited_mask = __ballot_sync(FULL_MASK, is_visited == false);
    const auto last_unvisited_pos = 32 - __clz(unvisited_mask);

    if (last_unvisited_pos) {
        float chosen_point;

        if (threadIdx.x + 1 == last_unvisited_pos) {
            const auto total = prefix_sum;
            const auto r = get_random_float(rng);
            chosen_point = min( total, total * r );
        }

        chosen_point = warp_bcast<float>(chosen_point, last_unvisited_pos - 1);

        const bool is_valid = (chosen_point <= prefix_sum);
        const auto chosen_pos = warp_vote(is_valid, unvisited_mask);

        assert(chosen_pos != 0);

        chosen_node = warp_bcast<uint32_t>(cand_node, chosen_pos - 1);
    }
    
    // Some unvisited node has to be selected:
    assert( is_visited || chosen_node != instance.dimension_ );

    return chosen_node;
}


/**
blockDim.x should equal instance.cand_list_size_
Supports blocks of size up to WARP_SIZE^2
*/
template<typename Tabu>
__device__
uint32_t block_roulette_choice_from_cand_list(
        InstanceContext instance,
        rand_state_t &rng,
        uint32_t current_node,
        float *cand_lists_product_cache,
        Tabu &tabu) {

    // One thread per candidate element
    assert(blockDim.x == instance.cand_list_size_);

    const auto offset = instance.cand_list_size_ * current_node;
    const auto cand_list = instance.cand_lists_ + offset;
    const auto cand_index = threadIdx.x;
    const auto cand_node = cand_list[cand_index];
    const bool is_visited = tabu.is_visited(cand_node);
    const auto product_cache = cand_lists_product_cache + offset;
    const auto product = is_visited ? 0 : product_cache[cand_index];

    uint32_t chosen_node = instance.dimension_;

    const auto prefix_sum = block_scan(product);  // Inclusive
    const auto last_unvisited_pos = block_vote_last( !is_visited );

    if (last_unvisited_pos) {
        float chosen_point;

        if (threadIdx.x + 1 == last_unvisited_pos) {
            const auto total = prefix_sum;
            const auto r = get_random_float(rng);
            chosen_point = min( total, total * r );
        }

        chosen_point = block_bcast(chosen_point, last_unvisited_pos - 1);

        const bool is_valid = (chosen_point <= prefix_sum && !is_visited);
        const auto chosen_pos = block_vote(is_valid);

        assert(chosen_pos != 0);

        chosen_node = block_bcast(cand_node, chosen_pos - 1);
    }
    // Some unvisited node has to be selected:
    assert( is_visited || chosen_node != instance.dimension_ );

    return chosen_node;
}


/**
Performs a block-level roulette wheel selection of a next node.
If no valid choice is possible it returns dimension.
The choice is made using the weighted reservoir sampling method by
Efraimidis and Spirakis

Only the thread with threadIdx.x is guaranteed to have a valid answer.
*/
template<typename Tabu>
__device__
uint32_t reservoir_sampling_roulette_choice_from_cand_list(
        InstanceContext instance,
        rand_state_t &rng,
        uint32_t current_node,
        float *cand_lists_product_cache,
        Tabu & tabu) {

    const auto offset = instance.cand_list_size_ * current_node;
    const auto cand_list = instance.cand_lists_ + offset;
    const auto product = cand_lists_product_cache + offset;

    uint32_t cand_node = instance.dimension_;
    float max_key = -FLT_MAX;

    for (auto index = threadIdx.x; index < instance.cand_list_size_; index += blockDim.x) {
        const auto node = cand_list[index];

        if ( tabu.is_available(node) ) {
            // We use max(..., FLT_MIN) to avoid computing log(0) for which
            // the result is -infinity
            const auto r = max(get_random_float(rng), FLT_MIN); 
            const auto key = __log2f(r) * product[index];
            if (key > max_key || cand_node == instance.dimension_) {
                cand_node = node;
                max_key = key;
            }
        }
    }
    return block_arg_max_atomics(cand_node, max_key);
}


__device__
float par_calculate_route_length(const InstanceContext &instance,
                                 uint32_t *route,
                                 char *shared_mem_buffer) {
    float my_sum = 0;
    for (uint32_t i = threadIdx.x; i < instance.dimension_; i += blockDim.x) {
        const auto node = route[i];
        const auto next_node = route[(i + 1) % instance.dimension_];
        my_sum += instance.get_distance(node, next_node);
    }
    // Now perform warp level reduce
    return block_sum(my_sum, shared_mem_buffer);
}


template<typename Tabu>
using roulette_choice_from_cand_list_fn = uint32_t (*)(
    InstanceContext /*instance*/,
    rand_state_t & /*rng*/,
    uint32_t /*current_node*/,
    float * /*nn_product_cache*/,
    Tabu & /*tabu*/);

/**
This constructs a complete solution to a problem and returns the route
in out_routes array.
*/
template<typename Tabu,
         roulette_choice_from_cand_list_fn<Tabu> roulette_choice_fn>
__global__
void build_ant_solution_using_cand_lists(
    InstanceContext instance,
    rand_state_t *rng_states,
    MMASRunContext ctx) {

    // This is a dynamically allocated shared memory for the threads in a block
    extern __shared__ char shared_mem[];
    __shared__ uint32_t chosen_node;

    const auto dimension = instance.dimension_;
    const auto ant_idx = blockIdx.x;
    uint32_t *ant_route = ctx.ant_routes_ + ant_idx * dimension;
    rand_state_t rng = rng_states[get_global_thread_id()];

    Tabu tabu(dimension, shared_mem, ant_route);
    tabu.parallel_init();
    char *shared_mem_buffer = shared_mem
                    + Tabu::get_required_shared_memory_size_in_bytes(dimension);

    if (threadIdx.x == 0) {
        chosen_node = get_random_uint32(rng, 0, dimension - 1);
    }
    __syncthreads();  // And now everyone has the updated chosen_node
    tabu.add_visited(chosen_node);

    auto current_node = chosen_node;

    for (uint32_t i = 1; i < dimension; ++i) {
        // Try to choose from the candidate list of current_node
        auto cand_node = roulette_choice_fn(
            instance,
            rng,
            current_node,
            ctx.cand_lists_product_cache_,
            tabu);

        assert( (cand_node == dimension) || tabu.is_available(cand_node) );

        if (cand_node == dimension) {  // All nearest neighbors were visited?
            // Choose among all the unvisited nodes the one with the maximum
            // product
            auto * const heuristic = instance.heuristic_matrix_ + current_node * dimension;
            auto * const pheromone = ctx.pheromone_matrix_ + current_node * dimension; 
            const auto length = tabu.get_length();
            float cand_product = -1;

            for (uint32_t i = threadIdx.x; i < length; i += blockDim.x) {
                const auto node = tabu.get_candidate(i);
                if (tabu.is_candidate_unvisited(node)) {
                    const float product = pheromone[node] * heuristic[node];
                    cand_product = max(cand_product, product);
                    cand_node = cand_product == product ? node : cand_node;
                }
            }
            cand_node = block_arg_max(cand_node, cand_product, shared_mem_buffer);
        }
        tabu.add_visited(cand_node);
        current_node = cand_node;
    }
    tabu.parallel_finalize();
    // Parallel calculation of the route length
    float my_sum = 0;
    for (uint32_t i = threadIdx.x; i < dimension; i += blockDim.x) {
        const auto node = ant_route[i];
        const auto next_node = ant_route[(i + 1) % dimension];
        my_sum += instance.distance_matrix_[node * dimension + next_node];
    }
    // Now perform warp level reduce
    my_sum = block_sum(my_sum, shared_mem_buffer);

    if (threadIdx.x == 0) {
        ctx.ant_route_costs_[ant_idx] = my_sum;
    }
    rng_states[get_global_thread_id()] = rng;  // Write back the updated PRNG's state
}


/**
Performs a block-level roulette wheel selection of a next node.
If no valid choice is possible it returns dimension.
The choice is made using the reservoir sampling method.

Only the thread with threadIdx.x == 0 is guaranteed to have a valid answer.
*/
template<typename Tabu>
__device__
uint32_t reservoir_sampling_roulette_choice(
        uint32_t dimension,
        rand_state_t &rng,
        uint32_t current_node,
        Tabu &tabu,
        float *product_cache,
        char * /* unused */) {

    const auto *product = product_cache + current_node * dimension;
    uint32_t cand_node = dimension;
    float max_key = -FLT_MAX;
    const auto nodes_to_visit_count = tabu.get_length();

    for (uint32_t i = threadIdx.x; i < nodes_to_visit_count; i += blockDim.x) {
        const auto node = tabu.get_candidate(i);
        if (tabu.is_candidate_unvisited(node)) {
            const auto r = max(get_random_float(rng), FLT_MIN);  // Assure that r > 0
            const auto key = __log2f(r) * product[node];
            if (key > max_key || cand_node == dimension) {
                cand_node = node;
                max_key = key;
            }
        }
    }
    return block_arg_max_atomics(cand_node, max_key);
}


__device__
uint32_t calc_chunk_size_per_thread(uint32_t n) {
    if (n <= blockDim.x) {
        return 1;
    }
    return n / blockDim.x + (n % blockDim.x != 0 ? 1 : 0);
}


template<typename Tabu>
using roulette_choice_fn = uint32_t (*)(
    uint32_t /*dimension*/,
    rand_state_t & /*rng*/,
    uint32_t /*current_node*/,
    Tabu & /*tabu*/,
    float * /*product_cache*/,
    char * /*shared_buffer*/);


template<typename Tabu>
__device__
uint32_t warp_roulette_choice(
        uint32_t dimension,
        rand_state_t &rng,
        uint32_t current_node,
        Tabu &tabu,
        float *product_cache,
        char * /*shared_buffer*/) {

    assert( get_warp_count() == 1 );

    auto * const product = product_cache + current_node * dimension;

    const auto nodes_to_visit_count = tabu.get_length();
    // Each thread gets a "chunk_size" of nodes and computes corresponding sum
    auto chunk_size  = calc_chunk_size_per_thread(nodes_to_visit_count);
    auto chunk_start = threadIdx.x * chunk_size;
    auto chunk_end   = min(chunk_start + chunk_size, nodes_to_visit_count);
    float chunk_sum = 0;
    uint32_t node = dimension;
    for (auto i = chunk_start; i < chunk_end; ++i) {
        const auto cand = tabu.get_candidate(i);
        if (tabu.is_candidate_unvisited(cand)) {
            node = cand;
            chunk_sum += product[node];
        }
    }

    auto prefix_sum = warp_scan(chunk_sum);
    float chosen_point;

    // The last thread draws a random number and multiplies it by the sum
    const auto last = min(WARP_SIZE, nodes_to_visit_count) - 1;
    if (threadIdx.x == last) {
        const auto r = get_random_float(rng);
        const auto total = prefix_sum;

        assert(total > 0);

        chosen_point = min(total, total * r);
    }

    chosen_point = warp_bcast(chosen_point, /* src thread: */ last);

    // Calculate position of the "chunk" containing the chosen point
    auto winner_pos = warp_vote(chosen_point <= prefix_sum && node != dimension);
    if (winner_pos == 0) {
        winner_pos = warp_vote(chosen_point > prefix_sum && node != dimension);
    }
    
    assert( winner_pos > 0 );

    auto winner_thread = winner_pos - 1;
    chunk_start = 0; 

    // If chunk_size > 1 then we need to find the corresponding node inside the
    // winner chunk of nodes
    while (chunk_size > 1) {
        chunk_start += winner_thread * chunk_size;
        chunk_end = min(chunk_start + chunk_size, nodes_to_visit_count);

        const auto start_sum = warp_bcast(max(0.0f, prefix_sum - chunk_sum),
                                          winner_thread);

        // Calculate the size of a sub-chunk for the thread
        chunk_size = calc_chunk_size_per_thread(chunk_size);

        // Calc. this thread's sub-chunk start & end
        const auto subchunk_start = chunk_start + threadIdx.x * chunk_size;
        const auto subchunk_end   = min(subchunk_start + chunk_size,
                                        nodes_to_visit_count);
        chunk_sum = 0;
        node = dimension;
        for (auto i = subchunk_start; i < subchunk_end; ++i) {
            const auto cand = tabu.get_candidate(i);
            if (tabu.is_candidate_unvisited(cand)) {
                node = cand;  // now we know that chunk contains an unvisited node
                chunk_sum += product[node];
            }
        }

        prefix_sum = start_sum + warp_scan(chunk_sum);

        winner_pos = warp_vote(chosen_point <= prefix_sum && node != dimension);
        // Due to a possible loss of precision when adding floats we need to 
        // assure the winner_pos is valid, i.e. contains an unvisited node
        if (winner_pos == 0) {
            winner_pos = warp_vote(chosen_point > prefix_sum && node != dimension);
        }
        winner_thread = winner_pos - 1;
    }
    return warp_bcast(node, winner_thread);
}


template<typename Tabu>
__device__
uint32_t block_roulette_choice(
        uint32_t dimension,
        rand_state_t &rng,
        uint32_t current_node,
        Tabu &tabu,
        float *product_cache,
        char * shared_buffer) {

    auto * const product = product_cache + current_node * dimension;

    const auto nodes_to_visit_count = tabu.get_length();
    // Each thread gets a "chunk_size" of nodes and computes corresponding sum
    auto chunk_size  = calc_chunk_size_per_thread(nodes_to_visit_count);
    auto chunk_start = threadIdx.x * chunk_size;
    auto chunk_end   = min(chunk_start + chunk_size, nodes_to_visit_count);
    float chunk_sum = 0;
    uint32_t node = dimension;
    for (auto i = chunk_start; i < chunk_end; ++i) {
        const auto cand = tabu.get_candidate(i);
        if (tabu.is_candidate_unvisited(cand)) {
            node = cand;
            chunk_sum += product[node];
        }
    }

    auto prefix_sum = block_scan(chunk_sum);
    float chosen_point;

    const auto last = min(blockDim.x, nodes_to_visit_count) - 1;
    if (threadIdx.x == last) {
        const auto r = get_random_float(rng);
        const auto total = prefix_sum;

        assert(total > 0);

        chosen_point = min(total, total * r);
    }

    chosen_point = block_bcast(chosen_point, /* src thread: */ last);

    // Calculate position of the "chunk" containing chosen_point
    auto winner_pos = block_vote(chosen_point <= prefix_sum && node != dimension);
    if (winner_pos == 0) {  // Due to the loss of precision when adding floats
        winner_pos = block_vote(chosen_point > prefix_sum && node != dimension);
    }
    
    assert( winner_pos > 0 );

    auto winner_thread = winner_pos - 1;
    chunk_start = 0; 

    // If chunk_size > 1 then we need to find the corresponding node inside the
    // winner chunk of nodes
    while (chunk_size > 1) {
        chunk_start += winner_thread * chunk_size;
        chunk_end = min(chunk_start + chunk_size, nodes_to_visit_count);

        const auto start_sum = block_bcast(max(0.0f, prefix_sum - chunk_sum),
                                           winner_thread);

        // Calculate the size of a sub-chunk for the thread
        chunk_size = calc_chunk_size_per_thread(chunk_size);

        // Calc. this thread's sub-chunk start & end
        const auto subchunk_start = chunk_start + threadIdx.x * chunk_size;
        const auto subchunk_end   = min(subchunk_start + chunk_size,
                                        nodes_to_visit_count);
        chunk_sum = 0;
        node = dimension;
        for (auto i = subchunk_start; i < subchunk_end; ++i) {
            const auto cand = tabu.get_candidate(i);
            if (tabu.is_candidate_unvisited(cand)) {
                node = cand;
                chunk_sum += product[node];
            }
        }

        prefix_sum = block_scan(chunk_sum) + start_sum;

        // Due to a possible loss of precision when adding floats we need to 
        // assure the winner_pos is in valid range
        winner_pos = block_vote(chosen_point <= prefix_sum && node != dimension);
        // Due to a possible loss of precision when adding floats we need to 
        // assure the winner_pos is valid, i.e. contains an unvisited node
        if (winner_pos == 0) {
            winner_pos = block_vote(chosen_point > prefix_sum && node != dimension);
        }
        winner_thread = winner_pos - 1;
    }
    return block_bcast(node, winner_thread);
}


/**
Constructs a complete solution to a TSP-like problem and returns the route
in out_routes array.

The construction is based on the MMAS algorithm without the nearest neighbors
lists.
*/
template<typename Tabu,
         roulette_choice_fn<Tabu> roulette_fn>
__global__
void build_ant_solution(
        InstanceContext instance,
        rand_state_t *rng_states,
        MMASRunContext ctx) {

    // This is a dynamically allocated shared memory for the threads in a block
    extern __shared__ char shared_mem[];
    __shared__ uint32_t chosen_node;

    const auto dimension = instance.dimension_;
    const auto ant_idx = blockIdx.x;
    uint32_t *ant_route = ctx.ant_routes_ + ant_idx * dimension;
    rand_state_t rng = rng_states[get_global_thread_id()];

    Tabu tabu(dimension, shared_mem, ant_route);
    tabu.parallel_init();

    char *shared_mem_buffer = shared_mem
      + Tabu::get_required_shared_memory_size_in_bytes(dimension);

    // Place an ant at a randomly chosen node
    if (threadIdx.x == 0) {
        chosen_node = get_random_uint32(rng, 0, dimension - 1);
    }  // Only thread 0 has the right value
    __syncthreads();  // And now everyone
    tabu.add_visited(chosen_node);

    auto current_node = chosen_node;
    for (uint32_t i = 1; i < dimension; ++i) {
        //const auto current_node = chosen_node;
        uint32_t cand_node = roulette_fn(
            dimension,
            rng,
            current_node,
            tabu,
            ctx.product_cache_,
            shared_mem_buffer
        );

        assert(cand_node < dimension);

        tabu.add_visited(cand_node);
        current_node = cand_node;
    }
    tabu.parallel_finalize();
    // Parallel calculation of the route length
    float my_sum = 0;
    for (uint32_t i = threadIdx.x; i < dimension; i += blockDim.x) {
        const auto node = ant_route[i];
        const auto next_node = ant_route[(i + 1) % dimension];
        my_sum += instance.distance_matrix_[node * dimension + next_node];
    }
    // Now perform warp level reduce
    my_sum = block_sum(my_sum, shared_mem_buffer);

    if (threadIdx.x == 0) {
        ctx.ant_route_costs_[ant_idx] = my_sum;
    }
    rng_states[get_global_thread_id()] = rng;  // Write back the updated PRNG's state
}


__device__
void update_best_solutions_log(MMASRunContext &ctx, float best_cost,
                               uint32_t iteration) {
    auto len = *ctx.global_best_log_length_;
    if (len < MMASRunContext::global_best_log_capacity) {
        auto &entry = ctx.global_best_log_[len++];
        entry.first = best_cost;
        entry.second = iteration;
        *ctx.global_best_log_length_ = len;
    }
}


// Assuming 1 block
__global__
void update_global_best_and_trail_limits(
        MMASRunContext ctx,
        uint32_t dimension,
        TrailLimits *trail_limits,
        MMASParameters params,
        uint32_t iteration) {

    assert(get_block_count() == 1);
    assert(get_warp_count() == 1);

    extern __shared__ char shared_mem[];

    auto *route_costs = ctx.ant_route_costs_;
    const auto route_count = ctx.ants_count_;
    float min_cost = route_costs[0];
    uint32_t iter_best_index = 0;

    for (uint32_t i = threadIdx.x; i < route_count; i += blockDim.x) {
        const auto cost = route_costs[i];
        if (min_cost > cost) {
            min_cost = cost;
            iter_best_index = i;
        }
    }

    __syncthreads();

    const auto best_route_idx = block_arg_max(iter_best_index, -min_cost,
                                              shared_mem);

    if (threadIdx.x == 0) {
        const auto best_cost = route_costs[best_route_idx];

        *ctx.iter_best_cost_ = best_cost;

        const auto *best_route = ctx.ant_routes_ + best_route_idx * dimension;
        // Create a copy of iter best route
        for (uint32_t i = 0; i < dimension; ++i) {
            ctx.iter_best_route_[i] = best_route[i];
        }

        if (best_cost < *ctx.global_best_cost_) {
            *ctx.global_best_cost_ = best_cost;

            for (uint32_t i = 0; i < dimension; ++i) {
                ctx.global_best_route_[i] = best_route[i];
            }
            update_best_solutions_log(ctx, best_cost, iteration);

            *trail_limits = calc_trail_limits(dimension,
                                              best_cost,
                                              params.get_evaporation_rate(),
                                              params.p_best_);
        }

        if (best_cost < *ctx.reset_best_cost_) {
            *ctx.reset_best_cost_ = best_cost;

            for (uint32_t i = 0; i < dimension; ++i) {
                ctx.reset_best_route_[i] = best_route[i];
            }
        }
    }
}


__global__
void zero_reset_route_cost(MMASRunContext ctx) {
    *ctx.reset_best_cost_ = FLT_MAX;
}


__global__
void init_rng_states(curandState_t *rngs, uint32_t dimension,
                     uint32_t seed) {

    const auto global_id = get_global_thread_id();
    curand_init(seed, global_id, /*offset: */0, &rngs[global_id]);
}


__global__
void evaporate_pheromone(uint32_t dimension,
                         float *pheromone,
                         float evaporation_rate,
                         TrailLimits *trail_limits,
                         bool reset) {

    assert(dimension == get_block_count());

    const auto node = blockIdx.x;
    const auto offset = node * dimension;
    const float min_trail = trail_limits->min_;

    for (uint32_t endpoint = threadIdx.x;
         endpoint < dimension;
         endpoint += blockDim.x) {

        if (reset) {
            pheromone[offset + endpoint] = trail_limits->max_;
        } else {
            const auto pher = pheromone[offset + endpoint];
            const auto updated = max(min_trail, pher * (1 - evaporation_rate));
            pheromone[offset + endpoint] = updated;
        }
    }
}


__global__
void deposit_pheromone(
        uint32_t dimension,
        float *pheromone,
        float *route_cost,
        uint32_t *route,
        TrailLimits *trail_limits,
        bool is_symmetric) {

    const auto deposit = 1 / *route_cost;
    const auto max_trail = trail_limits->max_;

    for (uint32_t i = threadIdx.x; i < dimension; i += blockDim.x) {
        auto node = route[i];
        auto next = route[(i + 1) % dimension];

        auto trail = pheromone[node * dimension + next];
        trail = fmin(max_trail, trail + deposit);

        pheromone[node * dimension + next] = trail;
        if (is_symmetric) {
            pheromone[next * dimension + node] = trail;
        }
    }
}


std::vector<float> create_heuristic_matrix(const ProblemInstance &instance,
                                           MMASParameters params) {
    const auto dim = instance.dimension_;
    std::vector<float> result(dim * dim);
    result.clear();
    for (auto i = 0u; i < dim; ++i) {
        for (auto j = 0u; j < dim; ++j) {
            const auto dist = instance.get_distance(i, j);
            const float h = dist > 0
                        ? static_cast<float>(1 / std::pow(dist, params.beta_))
                        : 1;
            result.push_back(h);
        }
    }
    return result;
}


TrailLimits calc_initial_trail_limits(
        ProblemInstance &instance,
        MMASParameters params,
        uint32_t start_node = 0) {

    const auto nn_sol = instance.create_solution_nn(start_node);
    const auto nn_cost = instance.calculate_route_length(nn_sol);
    
    std::cout << "NN sol. cost: " << nn_cost << std::endl;

    return calc_trail_limits(instance.dimension_,
                             nn_cost,
                             params.get_evaporation_rate(),
                             params.p_best_);
}


__device__
int32_t get_warp_lane_id() { return threadIdx.x % WARP_SIZE; }


struct NodeIndexPair {
    uint32_t node_;
    uint32_t index_;
};

/**
This method is used to find candidate nodes from which 2-opt moves can be
initiated. It should be called by all threads of a warp.
*/
__device__
NodeIndexPair two_opt_nn_find_candidate(
    uint32_t dimension,
    uint32_t *route,
    int8_t *dont_look_bits,
    uint32_t &warp_private_chunk_index,
    uint32_t &thread_private_node
) {
    NodeIndexPair candidate { dimension, 0 };

    while (warp_private_chunk_index < dimension
           && candidate.node_ == dimension) {

        auto warp_leader = warp_vote(thread_private_node != dimension);

        if (warp_leader != 0) {
            candidate.node_ = warp_bcast(thread_private_node, warp_leader - 1);
            candidate.index_ = warp_private_chunk_index
                             + warp_leader - 1;

            if (get_warp_lane_id() + 1 == warp_leader) {
                thread_private_node = dimension;
            }
        } else {
            warp_private_chunk_index += blockDim.x;

            auto i = warp_private_chunk_index + get_warp_lane_id();
            if (i < dimension) {
                auto node = route[i];
                thread_private_node = (dont_look_bits[node] == 0) ? node : dimension;
            }
        }
    }
    return candidate;
}


/*
Reverses a route segment given by beg & end, i.e. [beg, ..., end) (exclusive)
or equivalently reverses the other part of the route,
i.e. [0..beg)...[end..dimension).

The reversal is performed by all threads in a block (concurrently).
*/
__device__
bool reverse_route_segment(int32_t beg, int32_t end,
                           int32_t dimension,
                           uint32_t *route,
                           int32_t *pos_in_route) {

    int32_t len = end - beg + 1;  // length of the segment to reverse

    // Since we are dealing with a symmetric TSP we can either reverse
    // the specified segment of the route or reverse the rest of the route
    // instead. The decision is made based on the number of items that need
    // to be swapped.
    // Example: 
    //    - first case: a b [ c d e f ] g h => a b [ f e d c ] g h
    //    - sec. case:  a b [ c d e f ] g h => h g [ c d e f ] b a

    if (3 * len <= 2 * dimension) {  // The first case
        for (int32_t l = beg + threadIdx.x,
                     r = end - 1 - threadIdx.x;
            l < r;
            l += blockDim.x, r -= blockDim.x) {

            auto x = route[l];
            auto y = route[r];

            route[l] = y;
            route[r] = x;

            pos_in_route[x] = r;
            pos_in_route[y] = l;
        }
        __syncthreads();
        return true;
    } else {  // The second case
        --end;
        end = (end + 1 < dimension) ? end + 1 : 0;
        beg = (beg > 0) ? beg - 1 : dimension - 1;

        auto temp = end;
        end = beg;
        beg = temp;

        len = (beg <= end) ? (end - beg + 1) : (dimension - beg + end + 1);

        for (int32_t l = threadIdx.x,
                     r = len - 1 - threadIdx.x;
            l < r;
            l += blockDim.x, r -= blockDim.x) {

            auto xi = (beg + l) < dimension ? beg + l : beg + l - dimension;
            auto yi = (beg + r) < dimension ? beg + r : beg + r - dimension;

            auto x = route[xi];
            auto y = route[yi];

            route[xi] = y;
            route[yi] = x;

            pos_in_route[x] = yi;
            pos_in_route[y] = xi;
        }
    }
    return false;
}


/**
This is a parallel impl. of 2-opt heuristic for the (symmetric) TSP. It tries
to replace a pair of edges with another pair so that the total length of the
route is reduced.

The search for the edges is limited to the nearest neighbors of each node to
speed up the search. Additionaly, the "don't look bits" heuristic as proposed
by Bentely is also applied to speed up the search further at a possible
expense of a slightly longer route.
*/
__global__
void two_opt_nn(InstanceContext instance,
                uint32_t *all_routes,
                int32_t *all_pos_in_route,
                int8_t *all_dont_look_bits,
                float *all_route_costs) {

    // This is a dynamically allocated shared memory for the threads in a block
    extern __shared__ char shared_mem[];

    __shared__ uint32_t chosen_move_thread_id;
    __shared__ float chosen_move_gain;
    __shared__ bool any_candidate_nodes;
    __shared__ int32_t move_beg;
    __shared__ int32_t move_end;

    auto offset = blockIdx.x * instance.dimension_;
    uint32_t *route = all_routes + offset;
    int32_t *pos_in_route = all_pos_in_route + offset;
    int8_t *dont_look_bits = all_dont_look_bits + offset;
    float *route_cost = all_route_costs + blockIdx.x;

    for (uint32_t i = threadIdx.x; i < instance.dimension_; i += blockDim.x) {
        auto node = route[i];
        pos_in_route[node] = i;
        dont_look_bits[i] = 0;
    }

    if (threadIdx.x == 0) {
        chosen_move_thread_id = blockDim.x;  // We use blockDim.x as a sentinel
        chosen_move_gain = 0;
        any_candidate_nodes = false;
    }
    __syncthreads();

    const auto n = instance.dimension_;
    bool improvement_found = true;

    const auto Sentinel = instance.dimension_;
    uint32_t thread_private_node = Sentinel;

    while(improvement_found) {
        improvement_found = false;

        NodeIndexPair warp_cand { Sentinel, 0 };

        // Each thread processes its own chunk of the route's nodes
        uint32_t warp_private_chunk_index = get_warp_id() * WARP_SIZE;

        auto i = warp_private_chunk_index + get_warp_lane_id();
        
        const auto warp_leader = warp_vote(thread_private_node < Sentinel &&
                                           dont_look_bits[thread_private_node] == 0);

        if (warp_leader != 0) {
            warp_cand.node_ = warp_bcast(thread_private_node, warp_leader - 1);
            warp_cand.index_ = pos_in_route[warp_cand.node_];
        } 
        if (i < instance.dimension_) {
            auto node = route[i];
            thread_private_node = (dont_look_bits[node] == 0)
                                ? node
                                : Sentinel;
        }
        if (warp_cand.node_ == Sentinel) {
            const auto warp_leader = warp_vote(thread_private_node < Sentinel &&
                                               dont_look_bits[thread_private_node] == 0);
            if (warp_leader != 0) {
                warp_cand.node_ = warp_bcast(thread_private_node, warp_leader - 1);
                warp_cand.index_ = warp_private_chunk_index + warp_leader - 1;

                if (get_warp_lane_id() + 1 == warp_leader) {
                    thread_private_node = Sentinel;
                }
            }
        }
        while (true) {
            if (warp_cand.node_ == Sentinel) {
                warp_cand = two_opt_nn_find_candidate(
                    instance.dimension_,
                    route,
                    dont_look_bits,
                    warp_private_chunk_index,
                    thread_private_node);
            }

            // Warp private:
            float max_gain = 0;
            uint32_t left = 0;
            uint32_t right = 0;
            
            // Each thread in a warp checks one of the NN nodes 
            if (warp_cand.node_ != Sentinel) {
                const auto a = warp_cand.node_;
                i = warp_cand.index_;

                warp_cand.node_ = Sentinel;

                const auto a_succ = route[(i + 1 < n) ? i+1 : 0];
                const auto dist_a_to_succ = instance.get_distance(a, a_succ);

                const auto a_pred = route[(i > 0) ? i-1 : n-1];
                const auto dist_a_to_pred = instance.get_distance(a, a_pred);

                const auto cl_offset = instance.cand_list_size_ * a;
                const auto cand_list = instance.cand_lists_ + cl_offset;

                assert(instance.cand_list_size_ <= WARP_SIZE);

                const auto cl_i = get_warp_lane_id();

                if (cl_i < instance.cand_list_size_) {
                    const auto b = cand_list[cl_i];
                    const auto dist_ab = instance.get_distance(a, b);

                    const auto b_pos = pos_in_route[b];

                    if (dist_a_to_succ > dist_ab) {
                        auto b_succ = route[(b_pos + 1 < n) ? b_pos + 1 : 0];
                        auto diff = dist_a_to_succ
                                + instance.get_distance(b, b_succ)
                                - dist_ab
                                - instance.get_distance(a_succ, b_succ);

                        if (diff > max_gain) {
                            left = min(i, b_pos) + 1;
                            right = max(i, b_pos) + 1;
                            max_gain = diff;
                        }
                    }
                    if (dist_a_to_pred > dist_ab) {
                        auto b_pred = route[(b_pos > 0) ? b_pos-1 : n-1];
                        auto diff = dist_a_to_pred
                                + instance.get_distance(b_pred, b)
                                - dist_ab
                                - instance.get_distance(a_pred, b_pred);

                        if (diff > max_gain) {
                            left = min(i, b_pos);
                            right = max(i, b_pos);
                            max_gain = diff;
                        }
                    }
                }

                if (warp_vote(max_gain > 0) == 0) {  // No improvement possible
                    dont_look_bits[a] = 1;
                } else {  // At least one thread found a move with a positive gain
                    auto warp_max = warp_all_reduce_max(max_gain);
                    if (max_gain == warp_max && max_gain > 0) {
                        if (atomicMax(&chosen_move_gain, max_gain) < max_gain) {
                            chosen_move_thread_id = threadIdx.x;
                        }
                    }
                }
                if (get_warp_lane_id() == 0) {
                    any_candidate_nodes = true;
                }
            }

            __syncthreads();

            if (chosen_move_thread_id != blockDim.x) {  // Has anyone found a valid move?
                if (threadIdx.x == chosen_move_thread_id) {
                    move_beg = left;  // Winner shares the start and end of the
                    move_end = right; // route segment to reverse
                }
                __syncthreads();  // Sync. so that everyone has access to updated data
                left = move_beg;  // Read fresh value
                right = move_end;

                // Reverse route[left..right)
                reverse_route_segment(left, right,
                                      instance.dimension_,
                                      route, pos_in_route);

                if (threadIdx.x == 0) {
                    dont_look_bits[route[left]] = 0;
                    dont_look_bits[route[right-1]] = 0;

                    dont_look_bits[route[ (left > 0) ? left-1 : n-1 ]] = 0;
                    dont_look_bits[route[ (right < n) ? right : 0 ]] = 0;

                    // Clear the id of the winner so that the next move can be
                    // identified normally
                    chosen_move_thread_id = blockDim.x;
                    chosen_move_gain = 0;
                    any_candidate_nodes = false;
                }
                __syncthreads();

                improvement_found = true;

                break ;

            } else {
                const auto should_continue = any_candidate_nodes;

                __syncthreads();
                if (threadIdx.x == 0) {
                    any_candidate_nodes = false;
                }
                if ( !should_continue ) {
                    break ;
                }
            }
        }
    }
    __syncthreads();
    const auto final_cost = par_calculate_route_length(instance,
                                                       route, shared_mem);
    if (threadIdx.x == 0) {
        *route_cost = final_cost;
    }
}


using json = nlohmann::json;

double get_error_relative_to_optimum(const ProblemInstance &instance,
                                     double route_length) {
    const auto optimum = get_best_known_value(instance.name_, 0);
    
    if (optimum > 0) {
        return (route_length - optimum) / optimum;
    }
    return 0;
}

/**
Runs the MMAS with the provided arguments.{
Returns json instance with results and various measurements.
*/
template<typename Alg>
json run_gpu_based_mmas(ProblemInstance &instance,
                        MMASParameters &params,
                        Alg alg,
                        std::default_random_engine &rng,
                        uint32_t iterations,
                        uint32_t warps_per_block,
                        bool use_local_search,
                        uint32_t ls_warps_per_block) {

    using namespace std;

    Timer initialization_timer;

    const auto dimension = instance.dimension_;

    uniform_int_distribution<int32_t> distribution(0, instance.dimension_ - 1);
    uniform_real_distribution<> float_distribution(0.0, 1.0);
    const auto nn_sol_start_node = distribution(rng);
    const auto trail_limits = calc_initial_trail_limits(instance, params,
                                                        nn_sol_start_node);
    device_vector<TrailLimits> d_trail_limits(1, trail_limits);

    device_vector<float> d_pheromone(dimension * dimension, trail_limits.max_);
    device_vector<float> d_heuristic(create_heuristic_matrix(instance, params));

    // Convert vector of doubles to vector of floats
    vector<float> dist_matrix(dimension * dimension);
    {
        auto it = dist_matrix.begin();
        for (const double d : instance.distance_matrix_) {
            *it++ = static_cast<float>(d);
        }
    }
    device_vector<float> d_dist_matrix(dist_matrix);

    vector<float> coordinates;
    if (!instance.coordinates_.empty()) {
        coordinates.reserve(2 * dimension);
        for (auto p : instance.coordinates_) {
            coordinates.push_back(p.first);
            coordinates.push_back(p.second);
        }
    }
    device_vector<float> d_coordinates(coordinates);

    device_vector<float> d_cand_lists_product_cache(dimension * params.cand_list_size_);

    vector<uint32_t> cand_lists(dimension * params.cand_list_size_);
    cand_lists.clear();
    for (uint32_t node = 0; node < dimension; ++node) {
        for (const auto neighbor : instance.nearest_neighbor_lists_.at(node)) {
            cand_lists.push_back(neighbor);
        }
    }
    device_vector<uint32_t> d_cand_lists(cand_lists);
    device_vector<uint32_t> d_ant_routes(params.ants_count_ * dimension);
    device_vector<float>    d_route_costs(params.ants_count_,
                                          numeric_limits<float>::max());
    device_vector<int32_t> d_pos_in_routes(params.ants_count_ * dimension);
    device_vector<int8_t> d_dont_look_bits(params.ants_count_ * dimension);
    
    device_vector<uint32_t> d_temp_ant_routes(params.ants_count_ * dimension);

    const auto blocks_count = params.ants_count_;
    const uint32_t threads_per_block = warps_per_block * WARP_SIZE;

    device_vector<rand_state_t> d_rng_states(blocks_count * threads_per_block);

    const uint32_t seed = rng();
    init_rng_states<<<blocks_count, threads_per_block>>>(
        d_rng_states.data(),
        blocks_count * threads_per_block,
        seed);

    device_vector<float> d_iter_best_cost(1, numeric_limits<float>::max());
    device_vector<float> d_global_best_cost(1, numeric_limits<float>::max());
    device_vector<float> d_reset_best_cost(1, numeric_limits<float>::max());

    device_vector<uint32_t> d_iter_best_route(dimension);
    device_vector<uint32_t> d_global_best_route(dimension);
    device_vector<uint32_t> d_reset_best_route(dimension);

    device_vector<float> d_product_cache(dimension * dimension);
    device_vector<pair<float, uint32_t>> d_global_best_log(
        MMASRunContext::global_best_log_capacity);
    device_vector<uint32_t> d_global_best_log_length(1, 0);

    InstanceContext instance_ctx {
        dimension,
        d_coordinates,
        d_dist_matrix,
        d_heuristic,
        static_cast<float>(params.beta_),
        params.cand_list_size_,
        d_cand_lists
    };

    MMASRunContext mmas_ctx {
        params.ants_count_,
        d_pheromone,
        d_product_cache,
        d_cand_lists_product_cache,
        d_ant_routes,
        d_route_costs,
        d_iter_best_route,
        d_iter_best_cost,
        d_global_best_route,
        d_global_best_cost,
        d_reset_best_route,
        d_reset_best_cost,
        d_global_best_log,
        d_global_best_log_length
    };

    const auto initialization_time = initialization_timer.get_elapsed_seconds();
    Timer main_timer;

    GPUTimer build_sol_timer;
    GPUTimer evaporate_pheromone_timer;
    GPUTimer product_cache_update_timer;
    GPUTimer cand_list_product_cache_update_timer;
    GPUTimer update_best_timer;
    GPUTimer deposit_pheromone_timer;
    GPUTimer ls_timer;

    vector<pair<string, GPUTimer*>>timers{
        { "solution-build", &build_sol_timer },
        { "pheromone-evaporate", &evaporate_pheromone_timer },
        { "product-cache-update", &product_cache_update_timer },
        { "cand-list-product-cache-update", &cand_list_product_cache_update_timer },
        { "best-solution-update", &update_best_timer },
        { "pheromone-deposit", &deposit_pheromone_timer },
        { "local-search", &ls_timer }
    };

    uint32_t reset_iter = 0;
    float prev_best_cost = 0;
    int32_t stagnation_counter = 0;
    uint32_t stagnation_period = 500;  // The #iterations without a change to
                                       // the current reset best solution

    int max_active_num_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_num_blocks,
        alg.build_ant_sol_,
        alg.kernel_cfg_.threads_per_block_,
        alg.kernel_cfg_.shared_buffer_size_
    );
    cout << "Max active #blocks for ant sol. build kernel: "
         << max_active_num_blocks << endl;

    for (uint32_t iter = 0; iter < iterations; ++iter) {
        product_cache_update_timer.start();
        update_pheromone_heuristic_product_cache<<<dimension, 128>>>
            (instance_ctx, mmas_ctx, alg.use_product_reciprocal_);
        product_cache_update_timer.accumulate_elapsed();

        if (alg.use_cand_lists_) {
            cand_list_product_cache_update_timer.start();
            update_cand_lists_pheromone_heuristic_product_cache
                <<<dimension, 128>>>
                (instance_ctx, mmas_ctx, alg.use_product_reciprocal_);
            cand_list_product_cache_update_timer.accumulate_elapsed();
        }

        build_sol_timer.start();
        alg.build_ant_sol_<<<
                alg.kernel_cfg_.thread_blocks_,
                alg.kernel_cfg_.threads_per_block_,
                alg.kernel_cfg_.shared_buffer_size_
            >>>
            (instance_ctx, d_rng_states, mmas_ctx);
        build_sol_timer.accumulate_elapsed();

        if (use_local_search) {
            ls_timer.start();
            const int threads_per_block = ls_warps_per_block * WARP_SIZE;
            two_opt_nn<<<
                alg.kernel_cfg_.thread_blocks_,
                threads_per_block,
                sizeof(float) * ls_warps_per_block
            >>>(
                instance_ctx,
                d_ant_routes,
                d_pos_in_routes,
                d_dont_look_bits,
                mmas_ctx.ant_route_costs_
            );
            ls_timer.accumulate_elapsed();
        }

        update_best_timer.start();
        update_global_best_and_trail_limits<<<1, 32, 1 * (sizeof(uint32_t) + sizeof(float))>>>(
            mmas_ctx,
            dimension,
            d_trail_limits,
            params,
            iter);
        update_best_timer.accumulate_elapsed();

        if (iter % 10 == 0) {
            const auto iter_best_cost   = d_iter_best_cost.as_host_vector().front();
            const auto global_best_cost = d_global_best_cost.as_host_vector().front();
            const auto best_cost_since_reset  = d_reset_best_cost.as_host_vector().front();

            if (iter % 100 == 0) {
                cout << iter 
                    << ":\tGlobal | Reset | Iter. best costs: " 
                    << global_best_cost 
                    << " ("
                    << get_error_relative_to_optimum(instance, global_best_cost) * 100 << "%) \t"
                    << best_cost_since_reset << "\t"
                    << iter_best_cost << "\n";
            }

            if (prev_best_cost == best_cost_since_reset) {
                ++stagnation_counter;
            } else {
                stagnation_counter = 0;
                prev_best_cost = best_cost_since_reset;
            }
        }

        bool reset_pheromone = false;
        if (use_local_search
            && stagnation_counter * 10 >= stagnation_period
            && (iter - reset_iter) <= (iterations - iter)) {

            reset_pheromone = true;
            stagnation_counter = 0;
            reset_iter = iter;

            zero_reset_route_cost<<<1, 32>>>(mmas_ctx);
            cout << "Resetting pheromone at: " << reset_iter << "\n";
        }

        evaporate_pheromone_timer.start();
        evaporate_pheromone<<<dimension, 256>>>(
                dimension,
                d_pheromone,
                params.get_evaporation_rate(),
                d_trail_limits,
                reset_pheromone);
        evaporate_pheromone_timer.accumulate_elapsed();

        // If we do not use the local search then the pheromone is deposited
        // based on the current iteration best solution
        if ( !use_local_search ) {
            deposit_pheromone_timer.start();
            deposit_pheromone<<<1, 256>>>(
                dimension,
                d_pheromone,
                d_iter_best_cost,
                d_iter_best_route,
                d_trail_limits,
                true);
            deposit_pheromone_timer.accumulate_elapsed();
        } else if (!reset_pheromone) {
            // If we use the local search it is more effective to deposit
            // pheromone based on the current reset best or global best
            // solution.
            bool use_iter_best = float_distribution(rng) < 0.1;
            bool use_global_best = float_distribution(rng) < 0.1;
            deposit_pheromone_timer.start();
            deposit_pheromone<<<1, 256>>>(
                dimension,
                d_pheromone,
                use_iter_best ? d_iter_best_cost
                              : (use_global_best ? d_global_best_cost 
                                                 : d_reset_best_cost),
                use_iter_best ? d_iter_best_route
                              : (use_global_best ? d_global_best_route 
                                                 : d_reset_best_route),
                d_trail_limits,
                true);
            deposit_pheromone_timer.accumulate_elapsed();
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto elapsed_seconds = main_timer.get_elapsed_seconds();
    cout << "Elapsed (sec):" << elapsed_seconds << endl;

    const auto final_cost = static_cast<uint32_t>(d_global_best_cost.as_host_vector().front());
    const auto final_rel_err = get_error_relative_to_optimum(instance, final_cost);
    cout << "Final solution cost: " << final_cost
         << " (" << final_rel_err * 100 << "%)\n";

    const auto final_route = d_global_best_route.as_host_vector();
    if (!instance.is_route_valid(final_route)) {
        throw std::runtime_error("The final solution is invalid");
    }
    assert(static_cast<uint32_t>(instance.calculate_route_length(final_route)) == final_cost);

    cout << "Build sol. kernel mean time [ms]: " 
         << build_sol_timer.get_mean_elapsed_seconds() * 1000 << "\n";
    cout << "Evaporate pheromone mean time [ms]: "
         << evaporate_pheromone_timer.get_mean_elapsed_seconds() * 1000 << "\n";
    cout << "Heur. & pher. product cache mean update time [ms]: "
         << product_cache_update_timer.get_mean_elapsed_seconds() * 1000 << "\n";
    cout << "Cand list product cache mean update time [ms]: "
         << cand_list_product_cache_update_timer.get_mean_elapsed_seconds() * 1000 << "\n";
    cout << "Update best sol. mean time [ms]: "
         << update_best_timer.get_mean_elapsed_seconds() * 1000 << "\n";
    cout << "Deposit pheromone mean time [ms]: "
         << deposit_pheromone_timer.get_mean_elapsed_seconds() * 1000 << "\n";
    cout << "Local search time [ms]: "
         << ls_timer.get_mean_elapsed_seconds() * 1000 << "\n";

    json kernel_mean_times;
    float total_time_sec = 0;
    for (auto t : timers) {
        const auto time_sec = t.second->get_mean_elapsed_seconds();
        total_time_sec += t.second->get_total_elapsed_seconds();
        kernel_mean_times[t.first] = time_sec;
    }
    const auto mean_iter_time_sec = total_time_sec / iterations;
    cout << "Total time [s]: " << total_time_sec << "\n";
    cout << "Mean iter. time [ms]: " << mean_iter_time_sec * 1000 << "\n";

    auto final_route_json = json::array();
    for (auto node : final_route) {
        final_route_json.push_back(node);
    }

    auto best_solutions_log = json::array();
    const auto log_len = d_global_best_log_length.as_host_vector().front();
    const auto global_best_log = d_global_best_log.as_host_vector();
    for (uint32_t i = 0; i < log_len; ++i) {
        const auto &entry = global_best_log.at(i);
        best_solutions_log.push_back({ entry.first, entry.second });
    }

    json results = {
        { "total-time", elapsed_seconds },
        { "initialization-time", initialization_time },
        { "final-solution-cost", final_cost },
        { "final-solution-route", final_route_json },
        { "kernel-mean-times", kernel_mean_times },
        { "mean-iteration-time", mean_iter_time_sec },
        { "best-solutions-log", best_solutions_log },
        { "best-solution-error", final_rel_err }
    };
    return results;
}


template<typename args_t>
json cmdline_args_to_json(const args_t &args) {
    json result;
    for (auto &el : args) {
        const auto &val = el.second;
        if (val.isString()) {
            result[el.first] = val.asString();
        } else if (val.isBool()) {
            result[el.first] = val.asBool();
        } else if (val.isLong()) {
            result[el.first] = val.asLong();
        }
    }
    return result;
}


json to_json(const MMASParameters &params) {
    json result = {
        { "rho", params.rho_ },
        { "ants", params.ants_count_ },
        { "beta", params.beta_ },
        { "use-cand-lists", params.use_cand_lists_ },
        { "cand-list-size", params.cand_list_size_ },
        { "p-best", params.p_best_ }
    };
    return result;
}


json to_json(const ProblemInstance &instance) {
    json result = {
        { "name", instance.name_ },
        { "dimension", instance.dimension_ },
        { "is-symmetric", instance.is_symmetric_ }
    };
    return result;
}


std::string get_current_datetime_string(std::string datetime_sep = " ",
                                        std::string time_sep = ":") {
    using namespace std;

    auto t = time(0);   // get time now
    tm* now = localtime(&t);

    ostringstream out;

    out << (now->tm_year + 1900) << '-' 
        << (now->tm_mon + 1) << '-'
        << now->tm_mday << datetime_sep
        << now->tm_hour << time_sep
        << now->tm_min << time_sep
        << now->tm_sec;
    return out.str();
}


std::string get_results_filename(const ProblemInstance &instance,
                                 const std::string &alg_name) {
    using namespace std;
    ostringstream out;
    out << alg_name << '-'
        << instance.name_ << '_'
        << get_current_datetime_string("_", "_")
        << ".json";
    return out.str();
}


bool dir_exists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}


/**
 * Creates a list of directories as specified in the path.
 */
bool make_path(const std::string& path) {
    mode_t mode = 0755;
    int ret = mkdir(path.c_str(), mode);

    if (ret == 0) {
        return true;
    }

    switch (errno) {
    case ENOENT:
        // parent didn't exist, try to create it
        {
            const auto pos = path.find_last_of('/');
            if (pos == std::string::npos) {
                return false;
            }
            if (!make_path( path.substr(0, pos) )) {
                return false;
            }
        }
        // now, try to create again
        return 0 == mkdir(path.c_str(), mode);

    case EEXIST:
        // done!
        return dir_exists(path);
    }
    return false;
}


/**
 * Runs the MMAS based on the command line arguments passed in args
 */
void run_mmas_experiment(std::map<std::string, docopt::value> &args) {
    using namespace std;

    json experiment_log;

    experiment_log["experiment-started-at"] = get_current_datetime_string();
    experiment_log["arguments"] = cmdline_args_to_json(args);

    const auto path = args["--instance"].asString();

    auto instance = load_tsplib_instance(path.c_str());

    const auto dimension = instance.dimension_;
    const auto ants_count = args["--ants"].asLong() ? args["--ants"].asLong()
                                                    : dimension;

    MMASParameters params;
    params.ants_count_ = ants_count;

    const uint32_t warps_per_block = args["--block-warps"].asLong();

    const auto blocks_count = params.ants_count_;
    const uint32_t threads_per_block = warps_per_block * WARP_SIZE;

    struct KernelConfiguration {
        uint32_t thread_blocks_ = 0;
        uint32_t threads_per_block_ = 0;
        size_t shared_buffer_size_ = 0;
    };

    struct SolutionConstructionAlgorithm {
        typedef void (*build_ant_sol_fn)(InstanceContext ,
                                         rand_state_t *,
                                         MMASRunContext);

        build_ant_sol_fn build_ant_sol_;
        bool use_cand_lists_;
        bool use_product_reciprocal_;
        KernelConfiguration kernel_cfg_; 
    };
    const auto common_shared_mem_size = (warps_per_block + 1) * 2 * sizeof(float);

    /*
    List of available MMAS variants -- the names of the variables are created by
    joining the following acronyms describing the tabu list, node selection
    method and use of candidate lists.
     
    "lc" -- tabu impl. using the list compression by Uchida et al.
    "bt" -- tabu impl. using a bitmask, i.e. one bit per node
    "ct" -- compact tabu -- uses approx. half the memory required by the "lc"
    
    "rwm" -- Roulette Wheel Method
    "wrs" -- Weighted Reservoir Sampling

    "cl" -- variants of the MMAS using candidate lists to speed up the node
            selection process during solution construction; Lists of length 32
            are used by default.
    */
    SolutionConstructionAlgorithm mmas_rwm_lc_cl {
        (threads_per_block == WARP_SIZE)
            ? build_ant_solution_using_cand_lists<
                    CompressedListTabu,
                    warp_roulette_choice_from_cand_list<CompressedListTabu>
              >
            : build_ant_solution_using_cand_lists<
                    CompressedListTabu,
                    block_roulette_choice_from_cand_list<CompressedListTabu>
              >,
        /* use_cand_lists_ */ true,
        /* use_product_reciprocal_ */ false,
        {
            blocks_count,
            threads_per_block,
            CompressedListTabu::get_required_shared_memory_size_in_bytes(dimension)
                + common_shared_mem_size
        }
    };

    SolutionConstructionAlgorithm mmas_rwm_bt_cl {
        (threads_per_block == WARP_SIZE)
            ? build_ant_solution_using_cand_lists<
                    BitmaskTabu,
                    warp_roulette_choice_from_cand_list<BitmaskTabu>
              >
            : build_ant_solution_using_cand_lists<
                    BitmaskTabu,
                    block_roulette_choice_from_cand_list<BitmaskTabu>
              >,
        /* use_cand_lists_ */ true,
        /* use_product_reciprocal_ */ false,
        {
            blocks_count,
            threads_per_block,
            BitmaskTabu::get_required_shared_memory_size_in_bytes(dimension)
                + common_shared_mem_size
        }
    };

    SolutionConstructionAlgorithm mmas_rwm_ct_cl {
        (threads_per_block == WARP_SIZE)
            ? build_ant_solution_using_cand_lists<
                    CompactTabu,
                    warp_roulette_choice_from_cand_list<CompactTabu>
              >
            : build_ant_solution_using_cand_lists<
                    CompactTabu,
                    block_roulette_choice_from_cand_list<CompactTabu>
              >,
        /* use_cand_lists_ */ true,
        /* use_product_reciprocal_ */ false,
        {
            blocks_count,
            threads_per_block,
            CompactTabu::get_required_shared_memory_size_in_bytes(dimension)
                + common_shared_mem_size
        }
    };

    SolutionConstructionAlgorithm mmas_wrs_lc_cl {
        build_ant_solution_using_cand_lists<
            CompressedListTabu,
            reservoir_sampling_roulette_choice_from_cand_list<CompressedListTabu>
        >,
        /* use_cand_lists_ */ true,
        /* use_product_reciprocal_ */ true,
        {
            blocks_count,
            threads_per_block,
            CompressedListTabu::get_required_shared_memory_size_in_bytes(dimension)
                + common_shared_mem_size
        }
    };

    SolutionConstructionAlgorithm mmas_wrs_bt_cl {
        build_ant_solution_using_cand_lists<
            BitmaskTabu,
            reservoir_sampling_roulette_choice_from_cand_list<BitmaskTabu>
        >,
        /* use_cand_lists_ */ true,
        /* use_product_reciprocal_ */ true,
        {
            blocks_count,
            threads_per_block,
            BitmaskTabu::get_required_shared_memory_size_in_bytes(dimension)
                + common_shared_mem_size
        }
    };

    SolutionConstructionAlgorithm mmas_wrs_ct_cl {
        build_ant_solution_using_cand_lists<
            CompactTabu,
            reservoir_sampling_roulette_choice_from_cand_list<CompactTabu>
        >,
        /* use_cand_lists_ */ true,
        /* use_product_reciprocal_ */ true,
        {
            blocks_count,
            threads_per_block,
            CompactTabu::get_required_shared_memory_size_in_bytes(dimension)
                + common_shared_mem_size
        }
    };

    SolutionConstructionAlgorithm mmas_rwm_lc {
        (threads_per_block == WARP_SIZE)
            ? build_ant_solution<
                CompressedListTabu,
                warp_roulette_choice<CompressedListTabu>
              >
            : build_ant_solution<
                CompressedListTabu,
                block_roulette_choice<CompressedListTabu>
              >,
        /* use_cand_lists_ */ false,
        /* use_product_reciprocal_ */ false,
        {
            blocks_count,
            threads_per_block,
            CompressedListTabu::get_required_shared_memory_size_in_bytes(dimension)
                + max(warps_per_block + 1, WARP_SIZE) * 2 * sizeof(float)
        }
    };
    SolutionConstructionAlgorithm mmas_rwm_ct {
        (threads_per_block == WARP_SIZE)
            ? build_ant_solution<
                CompactTabu,
                warp_roulette_choice<CompactTabu>
              >
            : build_ant_solution<
                CompactTabu,
                block_roulette_choice<CompactTabu>
              >,
        /* use_cand_lists_ */ false,
        /* use_product_reciprocal_ */ false,
        {
            blocks_count,
            threads_per_block,
            CompactTabu::get_required_shared_memory_size_in_bytes(dimension)
                + max(warps_per_block + 1, WARP_SIZE) * 2 * sizeof(float)
        }
    };
    SolutionConstructionAlgorithm mmas_rwm_bt {
        (threads_per_block == WARP_SIZE)
            ? build_ant_solution<
                BitmaskTabu,
                warp_roulette_choice<BitmaskTabu>
              >
            : build_ant_solution<
                BitmaskTabu,
                block_roulette_choice<BitmaskTabu>
              >,
        /* use_cand_lists_ */ false,
        /* use_product_reciprocal_ */ false,
        {
            blocks_count,
            threads_per_block,
            BitmaskTabu::get_required_shared_memory_size_in_bytes(dimension)
                + max(warps_per_block + 1, WARP_SIZE) * 2 * sizeof(float)
        }
    };
    SolutionConstructionAlgorithm mmas_wrs_lc {
        build_ant_solution<
            CompressedListTabu,
            reservoir_sampling_roulette_choice<CompressedListTabu>
        >,
        /* use_cand_lists_ */ false,
        /* use_product_reciprocal_ */ true,
        {
            blocks_count,
            threads_per_block,
            CompressedListTabu::get_required_shared_memory_size_in_bytes(dimension)
                + common_shared_mem_size
        }
    };

    SolutionConstructionAlgorithm mmas_wrs_ct {
        build_ant_solution<
            CompactTabu,
            reservoir_sampling_roulette_choice<CompactTabu>
        >,
        /* use_cand_lists_ */ false,
        /* use_product_reciprocal_ */ true,
        {
            blocks_count,
            threads_per_block,
            CompactTabu::get_required_shared_memory_size_in_bytes(dimension)
                + common_shared_mem_size
        }
    };

    SolutionConstructionAlgorithm mmas_wrs_bt {
        build_ant_solution<
            BitmaskTabu,
            reservoir_sampling_roulette_choice<BitmaskTabu>
        >,
        /* use_cand_lists_ */ false,
        /* use_product_reciprocal_ */ true,
        {
            blocks_count,
            threads_per_block,
            BitmaskTabu::get_required_shared_memory_size_in_bytes(dimension)
                + common_shared_mem_size
        }
    };

    SolutionConstructionAlgorithm *alg_to_run { nullptr };
    const string alg_name = args["--alg"].asString();
    // This is a list of the MMAS variants that can be run
    if (alg_name == "mmas_rwm_lc") {  // Default, with the CompressedListTabu
        alg_to_run = &mmas_rwm_lc;
    } else if (alg_name == "mmas_rwm_ct") {
        alg_to_run = &mmas_rwm_ct;
    } else if (alg_name == "mmas_rwm_bt") {
        alg_to_run = &mmas_rwm_bt;
    } else if (alg_name == "mmas_wrs_lc") {
        alg_to_run = &mmas_wrs_lc;
    } else if (alg_name == "mmas_wrs_ct") {
        alg_to_run = &mmas_wrs_ct;
    } else if (alg_name == "mmas_wrs_bt") {
        alg_to_run = &mmas_wrs_bt;
    } else if (alg_name == "mmas_rwm_lc_cl") {
        alg_to_run = &mmas_rwm_lc_cl;
    } else if (alg_name == "mmas_rwm_bt_cl") {
        alg_to_run = &mmas_rwm_bt_cl;
    } else if (alg_name == "mmas_rwm_ct_cl") {
        alg_to_run = &mmas_rwm_ct_cl;
    } else if (alg_name == "mmas_wrs_lc_cl") {
        alg_to_run = &mmas_wrs_lc_cl;
    } else if (alg_name == "mmas_wrs_bt_cl") {
        alg_to_run = &mmas_wrs_bt_cl;
    } else if (alg_name == "mmas_wrs_ct_cl") {
        alg_to_run = &mmas_wrs_ct_cl;
    } else {
        cerr << "Unknown algorithm: " << alg_name << endl;
        abort();
    }

    assert(alg_to_run != nullptr);
    auto & alg = *alg_to_run;

    params.use_cand_lists_ = alg.use_cand_lists_;
    params.cand_list_size_ = args["--cand-list-size"].asLong();
    if (params.cand_list_size_ == 0
         || params.cand_list_size_ % WARP_SIZE != 0) {
        cerr << "Invalid candidates list size: " << params.cand_list_size_ << endl;
        abort();
    }

    instance.init_distance_matrix();
    instance.init_nn_lists(params.cand_list_size_);

    if (alg.use_cand_lists_) {
        if (warps_per_block * WARP_SIZE != params.cand_list_size_) {
            cerr << "# of threads in block should be equal to the cand. list"
                    " size." << endl;
            abort();
        }
    }

    params.rho_ = std::stof(args["--rho"].asString());

    uint32_t seed = args["--seed"].asLong();
    if (seed == 0) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    default_random_engine rng(seed);

    auto iter = args["--iter"].asLong();
    uint32_t iterations = iter;
    if (iterations == 0) {
        iterations = std::ceil(1000000.0 / params.ants_count_);
    }

    auto ls = args["--ls"].asLong();
    bool use_local_search = (ls == 1);

    uint32_t ls_warps_per_block = args["--ls-block-warps"].asLong();
    if (ls_warps_per_block == 0) {
        // This is a simple heuristic -- 1 warp per 10 nodes
        // It is best to set ls_warps_per_block manually depending on the GPU
        // used
        ls_warps_per_block = static_cast<uint32_t>(ceil(dimension / (10 * 32)));
        ls_warps_per_block = std::max(1u, std::min(32u, ls_warps_per_block));
    }
    experiment_log["ls-block-warps"] = ls_warps_per_block;

    auto trials_log = json::array();
    const auto trials = args["--trials"].asLong();
    for (auto trial = 0; trial < trials; ++trial) {
        cout << "\nStarting trial " << trial << "\n\n";
        trials_log.emplace_back(
            run_gpu_based_mmas(instance, params, alg, rng,
                               iterations,
                               warps_per_block,
                               use_local_search,
                               ls_warps_per_block) );
    }
    experiment_log["rng-seed"] = seed;
    experiment_log["algorithm"] = alg_name;
    experiment_log["instance"] = to_json(instance);
    experiment_log["iterations-count"] = iterations;
    experiment_log["trials-count"] = trials;
    experiment_log["mmas-parameters"] = to_json(params);
    experiment_log["experiment-finished-at"] = get_current_datetime_string();
    experiment_log["trials"] = trials_log;

    const auto results_dir = args["--results-dir"].asString();
    // Linux is assumed
    make_path(results_dir);
    const auto results_path = results_dir + "/"
                            + get_results_filename(instance, alg_name);

    cout << "Saving results to: " << results_path << endl;
    ofstream out(results_path);
    if (out.is_open()) {
        out << experiment_log.dump(2);
        out.close();
    }
}
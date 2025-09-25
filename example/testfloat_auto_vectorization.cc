/**
 *  @file benchmark_chunks.cc
 *  @author Jiefeng Zhou
 *  @date Nov, 2025
 *  @brief A microbenchmark tool for testing specific SZP single-threaded buffer-to-buffer functions
 *         using pre-chunked binary data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <string>
#include <cmath> // For sqrt
#include <cfloat> // For FLT_MAX
#include "szp.h"


// --- Timing Utility ---

double totalCost = 0;
struct timeval costStart;

void cost_start() {
    gettimeofday(&costStart, NULL);
}

void cost_end() {
    struct timeval costEnd;
    gettimeofday(&costEnd, NULL);
    totalCost = ((costEnd.tv_sec * 1000000 + costEnd.tv_usec) - (costStart.tv_sec * 1000000 + costStart.tv_usec)) / 1000000.0;
}

// --- Statistics Structure and Calculation ---

struct StatsResult {
    double min_time;
    double max_time;
    double avg_time;
    double std_dev;
    double throughput_mbs;
};

StatsResult calculate_stats(const std::vector<double>& timings, size_t data_size_bytes) {
    StatsResult stats = {0};
    if (timings.empty()) return stats;

    std::vector<double> sorted_timings = timings;
    std::sort(sorted_timings.begin(), sorted_timings.end());

    stats.min_time = sorted_timings.front();
    stats.max_time = sorted_timings.back();

    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    stats.avg_time = sum / timings.size();

    double sq_sum = 0.0;
    for (const auto& t : timings) {
        sq_sum += (t - stats.avg_time) * (t - stats.avg_time);
    }
    stats.std_dev = sqrt(sq_sum / timings.size());

    double data_size_mb = static_cast<double>(data_size_bytes) / (1024 * 1024);
    if (stats.avg_time > 0) {
        stats.throughput_mbs = data_size_mb / stats.avg_time;
    }

    return stats;
}

// --- File I/O and Data Analysis Utilities ---

/**
 * @brief Reads a chunk of float data from a file, auto-detecting format.
 * 
 * Supports .npy (skips 128-byte header) and raw formats like .f32, .dat
 * (reads from the beginning).
 */
float* read_float_data(const char* filepath, size_t num_elements) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open input file %s\n", filepath);
        return nullptr;
    }

    // Check file extension to handle .npy header
    const char* dot = strrchr(filepath, '.');
    if (dot && strcmp(dot, ".npy") == 0) {
        const int NPY_HEADER_SIZE = 128;
        fseek(file, NPY_HEADER_SIZE, SEEK_SET);
        printf("Detected .npy format, skipping %d-byte header.\n", NPY_HEADER_SIZE);
    } else {
        printf("Detected raw data format (.f32, .dat, etc.), reading from beginning.\n");
    }

    float* data = (float*)malloc(num_elements * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed for reading data file.\n");
        fclose(file);
        return nullptr;
    }

    size_t elements_read = fread(data, sizeof(float), num_elements, file);
    if (elements_read != num_elements) {
        fprintf(stderr, "Warning: Expected to read %zu floats, but only got %zu from %s\n",
                num_elements, elements_read, filepath);
    }
    fclose(file);
    return data;
}

/**
 * @brief Calculates and prints basic statistics of the raw input data.
 */
void calculate_and_print_data_stats(const float* data, size_t num_elements) {
    if (!data || num_elements == 0) return;

    printf("  --- Raw Data Statistics ---\n");
    
    // First 10 elements
    printf("  First ~10 elements: [");
    size_t preview_count = std::min((size_t)10, num_elements);
    for (size_t i = 0; i < preview_count; ++i) {
        printf("%.4f%s", data[i], (i == preview_count - 1) ? "" : ", ");
    }
    printf("]\n");

    // Min, Max, Range
    float min_val = data[0];
    float max_val = data[0];
    double sum = 0.0;

    for (size_t i = 0; i < num_elements; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        sum += data[i];
    }
    double range = max_val - min_val;
    
    // Standard Deviation
    double mean = sum / num_elements;
    double sq_diff_sum = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
        sq_diff_sum += (data[i] - mean) * (data[i] - mean);
    }
    double std_dev = sqrt(sq_diff_sum / num_elements);

    printf("  Min: %-15.6f | Max: %-15.6f | Range: %-15.6f | Std Dev: %.6f\n",
           min_val, max_val, range, std_dev);
}


// --- Main Benchmark Logic ---

void print_usage() {
    printf("Usage: ./benchmark_multidim <data_file.[npy|f32|dat]> <err_bound> <num_chunks> <algo_block_size> [-r reps] [-w warmups]\n");
    printf("\n  <data_file>       : Path to the data file (.npy, .f32, .dat supported).\n");
    printf("  <err_bound>       : Absolute error bound for compression (e.g., 1e-4).\n");
    printf("  <num_chunks>      : Number of chunks to test in a batch (e.g., 512).\n");
    printf("  <algo_block_size> : Internal block size for the SZP algorithm (e.g., 1024).\n");
    printf("  -r <reps>         : Number of repetitions for measurement (default: 20).\n");
    printf("  -w <warmups>      : Number of warmup runs (default: 5).\n\n");
    printf("Example:\n");
    printf("  ./benchmark_multidim ../node_feat.f32 1e-4 512 1024 -r 50 -w 10\n");
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        print_usage();
        return 1;
    }

    const char* data_file_path = argv[1];
    float abs_err_bound = atof(argv[2]);
    int num_chunks_per_test = atoi(argv[3]);
    int algo_block_size = atoi(argv[4]);
    int repetitions = 20;
    int warmup_runs = 5;

    for (int i = 5; i < argc; ++i) {
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) repetitions = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) warmup_runs = atoi(argv[++i]);
    }

    printf("--- SZP Single-Thread Benchmark (Buffer vs. Non-Buffer) ---\n");
    printf("Source Data: %s\n", data_file_path);
    printf("Error Bound: %.2e\n", abs_err_bound);
    printf("Config: %d warmups, %d repetitions\n", warmup_runs, repetitions);
    printf("========================================================================================\n");

    const std::vector<size_t> chunk_sizes_bytes = {512, 1024, 2048, 4096, 8192, 16384};

    for (const auto& chunk_size : chunk_sizes_bytes) {
        size_t total_data_bytes = num_chunks_per_test * chunk_size;
        size_t num_elements = total_data_bytes / sizeof(float);
        
        printf("\n--- Test Case: %d chunks of %zu Bytes (Total: %.2f MB), Algo Block Size: %d ---\n",
               num_chunks_per_test, chunk_size, (double)total_data_bytes / (1024*1024), algo_block_size);

        if (num_elements == 0) {
            printf("  Skipping: Total data size is too small to contain a single float.\n");
            continue;
        }

        float* original_data = read_float_data(data_file_path, num_elements);
        if (!original_data) continue;

        // Print statistics of the loaded raw data
        calculate_and_print_data_stats(original_data, num_elements);

        // Allocate separate, isolated buffers for each version
        size_t max_compressed_size = total_data_bytes + 1024;
        unsigned char* compressed_data_nb = (unsigned char*)malloc(max_compressed_size);
        float* decompressed_data_nb = (float*)malloc(total_data_bytes);
        unsigned char* compressed_data_b = (unsigned char*)malloc(max_compressed_size);
        float* decompressed_data_b = (float*)malloc(total_data_bytes);

        // Pre-allocated temp buffers are only for the "_buffer" functions
        unsigned char* temp_sign_arr = (unsigned char*)malloc(num_elements * sizeof(unsigned char));
        unsigned int* temp_predict_arr = (unsigned int*)malloc(num_elements * sizeof(unsigned int));
        int* temp_quant_arr = (int*)malloc(num_elements * sizeof(int));
        
        if (!compressed_data_nb || !decompressed_data_nb || !compressed_data_b || !decompressed_data_b || !temp_sign_arr || !temp_predict_arr || !temp_quant_arr) {
            fprintf(stderr, "Error: Failed to allocate one or more memory buffers. Skipping.\n");
            free(original_data); free(compressed_data_nb); free(decompressed_data_nb);
            free(compressed_data_b); free(decompressed_data_b);
            free(temp_sign_arr); free(temp_predict_arr); free(temp_quant_arr);
            continue;
        }
        
        size_t compressed_size_nb = 0, compressed_size_b = 0;

        // --- Run Non-Buffer Benchmarks ---
        std::vector<double> comp_timings_nb;
        for (int i = 0; i < warmup_runs + repetitions; ++i) {
            cost_start();
            szp_float_single_thread_arg(compressed_data_nb, original_data, &compressed_size_nb, abs_err_bound, num_elements, algo_block_size);
            cost_end();
            if (i >= warmup_runs) comp_timings_nb.push_back(totalCost);
        }
        StatsResult comp_stats_nb = calculate_stats(comp_timings_nb, total_data_bytes);

        std::vector<double> decomp_timings_nb;
        for (int i = 0; i < warmup_runs + repetitions; ++i) {
            cost_start();
            szp_float_decompress_single_thread_arg(decompressed_data_nb, num_elements, abs_err_bound, algo_block_size, compressed_data_nb + sizeof(float));
            cost_end();
            if (i >= warmup_runs) decomp_timings_nb.push_back(totalCost);
        }
        StatsResult decomp_stats_nb = calculate_stats(decomp_timings_nb, total_data_bytes);

        // --- Run Buffer Benchmarks ---
        std::vector<double> comp_timings_b;
        for (int i = 0; i < warmup_runs + repetitions; ++i) {
            cost_start();
            compressed_size_b = szp_float_single_thread_arg_buffer(compressed_data_b, original_data, abs_err_bound, num_elements, temp_sign_arr, temp_predict_arr, temp_quant_arr);
            cost_end();
            if (i >= warmup_runs) comp_timings_b.push_back(totalCost);
        }
        StatsResult comp_stats_b = calculate_stats(comp_timings_b, total_data_bytes);

        std::vector<double> decomp_timings_b;
        for (int i = 0; i < warmup_runs + repetitions; ++i) {
            cost_start();
            szp_float_decompress_single_thread_arg_buffer(decompressed_data_b, num_elements, abs_err_bound, compressed_data_b + sizeof(float), temp_sign_arr, temp_predict_arr);
            cost_end();
            if (i >= warmup_runs) decomp_timings_b.push_back(totalCost);
        }
        StatsResult decomp_stats_b = calculate_stats(decomp_timings_b, total_data_bytes);

        // --- Calculate Derived Metrics & Perform Separate Verifications ---
        double comp_ratio_nb = (compressed_size_nb > 0) ? (double)total_data_bytes / compressed_size_nb : 0;
        double comp_ratio_b = (compressed_size_b > 0) ? (double)total_data_bytes / compressed_size_b : 0;
        double comp_time_speedup = (comp_stats_b.avg_time > 0) ? comp_stats_nb.avg_time / comp_stats_b.avg_time : 0;
        double decomp_time_speedup = (decomp_stats_b.avg_time > 0) ? decomp_stats_nb.avg_time / decomp_stats_b.avg_time : 0;
        
        double max_abs_err_nb = 0.0;
        for (size_t i = 0; i < num_elements; ++i) {
            max_abs_err_nb = std::max(max_abs_err_nb, (double)fabs(original_data[i] - decompressed_data_nb[i]));
        }
        double max_abs_err_b = 0.0;
        for (size_t i = 0; i < num_elements; ++i) {
            max_abs_err_b = std::max(max_abs_err_b, (double)fabs(original_data[i] - decompressed_data_b[i]));
        }

        // --- Print Unified Table ---
        printf("  %-20s | %18s | %18s | %s\n", "Metric", "Non-Buffer Version", "Buffer Version", "Notes");
        printf("  ---------------------|--------------------|--------------------|-------------------------\n");

        printf("  -- Compression --\n");
        printf("  %-20s | %18.6f | %18.6f | Speedup: %.2fx\n", "Avg Time (s)", comp_stats_nb.avg_time, comp_stats_b.avg_time, comp_time_speedup);
        printf("  %-20s | %18.6f | %18.6f | \n", "Std Dev (s)", comp_stats_nb.std_dev, comp_stats_b.std_dev);
        printf("  %-20s | %18.2f | %18.2f | \n", "Throughput (MB/s)", comp_stats_nb.throughput_mbs, comp_stats_b.throughput_mbs);
        printf("  %-20s | %18.2f | %18.2f | (Ratio to 1)\n", "Compression Ratio", comp_ratio_nb, comp_ratio_b);

        printf("  -- Decompression --\n");
        printf("  %-20s | %18.6f | %18.6f | Speedup: %.2fx\n", "Avg Time (s)", decomp_stats_nb.avg_time, decomp_stats_b.avg_time, decomp_time_speedup);
        printf("  %-20s | %18.6f | %18.6f | \n", "Std Dev (s)", decomp_stats_nb.std_dev, decomp_stats_b.std_dev);
        printf("  %-20s | %18.2f | %18.2f | \n", "Throughput (MB/s)", decomp_stats_nb.throughput_mbs, decomp_stats_b.throughput_mbs);

        printf("  -- Verification --\n");
        printf("  %-20s | %18.6e | %18.6e | (Bound: %.2e)\n", "Max Abs Error", max_abs_err_nb, max_abs_err_b, abs_err_bound);


        // Cleanup all allocated memory
        free(original_data);
        free(compressed_data_nb);
        free(decompressed_data_nb);
        free(compressed_data_b);
        free(decompressed_data_b);
        free(temp_sign_arr);
        free(temp_predict_arr);
        free(temp_quant_arr);
    }

    printf("\n========================================================================================\n");
    printf("Benchmark complete.\n");
    return 0;
}
// ./example/test_chunk_benchmark /anvil/projects/x-cis240192/x-jzhou28/data/igb_datasets/tiny/processed/paper/node_feat.npy 3e-3 -r 50 -w 10
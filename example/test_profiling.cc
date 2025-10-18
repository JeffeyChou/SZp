/**
 *  @file test_profiling.cc
 *  @author Jiefeng Zhou
 *  @date Oct, 2025
 *  @brief An example to test the profiling functions in szp_profiling.cc.
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
#include "szp_profiling.h"

// --- File I/O Utility ---

float* read_float_data(const char* filepath, size_t num_elements) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open input file %s\n", filepath);
        return nullptr;
    }

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

// --- Statistics Calculation & Reporting ---

// Helper to format large numbers with commas for readability
std::string format_with_commas(size_t n) {
    std::string s = std::to_string(n);
    int len = s.length();
    int commas = (len - 1) / 3;
    if (commas == 0) return s;
    std::string result = "";
    result.reserve(len + commas);
    int first_part_len = len % 3;
    if (first_part_len == 0) first_part_len = 3;
    result += s.substr(0, first_part_len);
    for (int i = first_part_len; i < len; i += 3) {
        result += ',';
        result += s.substr(i, 3);
    }
    return result;
}

double calculate_trimmed_mean(std::vector<double>& data) {
    if (data.empty()) return 0.0;
    if (data.size() < 4) { // Not enough data to trim
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }
    std::sort(data.begin(), data.end());
    size_t trim_count = std::max((size_t)1, (size_t)(data.size() * 0.1)); // Trim 10% from each end

    double sum = 0.0;
    for (size_t i = trim_count; i < data.size() - trim_count; ++i) {
        sum += data[i];
    }
    return sum / (data.size() - 2 * trim_count);
}

void print_statistics(const std::string& name, std::vector<struct szp_prof_data>& results, size_t num_elements, bool is_compression, bool is_omp) {
    if (results.empty()) return;

    std::vector<double> total_times, alloc_times;
    std::vector<double> t1, t2, t3, t4;
    std::vector<double> ops1, ops4;
    double ratio1 = 0.0, ratio4 = 0.0;

    for (const auto& res : results) {
        total_times.push_back(res.total);
        alloc_times.push_back(res.allocation);
        if (is_compression) {
            t1.push_back(res.t_quant);
            t2.push_back(res.t_pred);
            t3.push_back(res.t_sign_abs_max);
            t4.push_back(res.t_packing);
            ops4.push_back(static_cast<double>(res.ops_packing));
        } else { // Decompression
            t1.push_back(res.t_unpacking);
            t2.push_back(res.t_sign_restore);
            t3.push_back(res.t_pred_recon);
            t4.push_back(res.t_dequant);
            ops1.push_back(static_cast<double>(res.ops_unpacking));
        }
    }

    double avg_total = calculate_trimmed_mean(total_times);
    double avg_alloc = calculate_trimmed_mean(alloc_times);
    double avg_t1 = calculate_trimmed_mean(t1);
    double avg_t2 = calculate_trimmed_mean(t2);
    double avg_t3 = calculate_trimmed_mean(t3);
    double avg_t4 = calculate_trimmed_mean(t4);
    size_t avg_ops1 = static_cast<size_t>(calculate_trimmed_mean(ops1));
    size_t avg_ops4 = static_cast<size_t>(calculate_trimmed_mean(ops4));

    ratio1 = static_cast<double>(avg_ops1) / num_elements;
    ratio4 = static_cast<double>(avg_ops4) / num_elements;


    printf("\n--- Averaged Statistics for %s ---\n", name.c_str());

    if (is_omp) {
        printf("  - %-25s %.6f s\n", "Total Wall-Clock Time:", avg_total);
        printf("  - %-25s %.6f s | \n", "Allocation Time:", avg_alloc);
        printf("  - Sub-stage timings are Total CPU Time (sum across all threads):\n");
        if (is_compression) {
            printf("    - %-23s %.6f s | \n", "Quantization:", avg_t1);
            printf("    - %-23s %.6f s | \n", "Prediction:", avg_t2);
            printf("    - %-23s %.6f s | \n", "Sign/Abs/Max:", avg_t3);
            printf("    - %-23s %.6f s | Ops: %-18s (Ratio: %.2f)\n", "Packing:", avg_t4, format_with_commas(avg_ops4).c_str(), ratio4);
        } else {
            printf("    - %-23s %.6f s | Ops: %-18s (Ratio: %.2f)\n", "Unpacking:", avg_t1, format_with_commas(avg_ops1).c_str(), ratio1);
            printf("    - %-23s %.6f s | \n", "Sign Restoration:", avg_t2);
            printf("    - %-23s %.6f s | \n", "Prediction Recon.:", avg_t3);
            printf("    - %-23s %.6f s | \n", "Dequantization:", avg_t4);
        }
    } else { // Single-threaded
        double total_compute = avg_t1 + avg_t2 + avg_t3 + avg_t4;
        printf("  - %-25s %.6f s\n", "Total Time:", avg_total);
        printf("    - %-23s %.6f s (%5.2f%%) |\n", "Allocation:", avg_alloc, avg_total > 0 ? (avg_alloc / avg_total * 100) : 0);
        if (is_compression) {
            printf("    - %-23s %.6f s (%5.2f%%) |\n", "Quantization:", avg_t1, total_compute > 0 ? (avg_t1 / total_compute * 100) : 0);
            printf("    - %-23s %.6f s (%5.2f%%) |\n", "Prediction:", avg_t2, total_compute > 0 ? (avg_t2 / total_compute * 100) : 0);
            printf("    - %-23s %.6f s (%5.2f%%) |\n", "Sign/Abs/Max:", avg_t3, total_compute > 0 ? (avg_t3 / total_compute * 100) : 0);
            printf("    - %-23s %.6f s (%5.2f%%) | Ops: %-18s (Ratio: %.2f)\n", "Packing:", avg_t4, total_compute > 0 ? (avg_t4 / total_compute * 100) : 0, format_with_commas(avg_ops4).c_str(), ratio4);

        } else {
            printf("    - %-23s %.6f s (%5.2f%%) | Ops: %-18s (Ratio: %.2f)\n", "Unpacking:", avg_t1, total_compute > 0 ? (avg_t1 / total_compute * 100) : 0, format_with_commas(avg_ops1).c_str(), ratio1);
            printf("    - %-23s %.6f s (%5.2f%%) |\n", "Sign Restoration:", avg_t2, total_compute > 0 ? (avg_t2 / total_compute * 100) : 0);
            printf("    - %-23s %.6f s (%5.2f%%) |\n", "Prediction Recon.:", avg_t3, total_compute > 0 ? (avg_t3 / total_compute * 100) : 0);
            printf("    - %-23s %.6f s (%5.2f%%) |\n", "Dequantization:", avg_t4, total_compute > 0 ? (avg_t4 / total_compute * 100) : 0);
        }
    }
}

// --- Main Test Logic ---

void print_usage() {
    printf("Usage: ./test_profiling <data_file> <num_elements> <err_bound> <algo_block_size> [-w warmups] [-r reps]\n");
    printf("\n  <data_file>       : Path to the data file (.npy, .f32, .dat supported).\n");
    printf("  <num_elements>    : Number of elements to read from the data file.\n");
    printf("  <err_bound>       : Absolute error bound for compression (e.g., 1e-4).\n");
    printf("  <algo_block_size> : Internal block size for the SZP algorithm (e.g., 1024).\n");
    printf("  -w <warmups>      : Number of warmup runs (default: 10).\n");
    printf("  -r <reps>         : Number of repetitions for measurement (default: 30).\n");
    printf("\nExample:\n");
    printf("  ./test_profiling ../node_feat.f32 100000 1e-4 1024 -w 5 -r 20\n");
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        print_usage();
        return 1;
    }

    const char* data_file_path = argv[1];
    size_t num_elements = atoi(argv[2]);
    float abs_err_bound = atof(argv[3]);
    int algo_block_size = atoi(argv[4]);
    int warmup_runs = 10;
    int repetitions = 30;

    for (int i = 5; i < argc; ++i) {
        if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) warmup_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) repetitions = atoi(argv[++i]);
    }

    printf("--- SZP Profiling Test ---\n");
    printf("Data File: %s\n", data_file_path);
    printf("Elements: %s\n", format_with_commas(num_elements).c_str());
    printf("Error Bound: %.2e\n", abs_err_bound);
    printf("Algorithm Block Size: %d\n", algo_block_size);
    printf("Warmup Runs: %d\n", warmup_runs);
    printf("Repetitions: %d\n", repetitions);
    printf("========================================================================================\n");

    float* original_data = read_float_data(data_file_path, num_elements);
    if (!original_data) return 1;

    size_t compressed_size = 0;
    size_t max_compressed_size = num_elements * sizeof(float) + 1024;
    unsigned char* compressed_data = (unsigned char*)malloc(max_compressed_size);
    float* decompressed_data = (float*)malloc(num_elements * sizeof(float));

    if (!compressed_data || !decompressed_data) {
        fprintf(stderr, "Error: Failed to allocate memory for compression/decompression buffers.\n");
        free(original_data);
        return 1;
    }

    // --- Warmup Phase ---
    printf("\n--- Running %d Warmup Iterations ---\n", warmup_runs);
    for (int i = 0; i < warmup_runs; ++i) {
        // Original versions
        szp_float_single_thread_arg_profile(compressed_data, original_data, &compressed_size, abs_err_bound, num_elements, algo_block_size);
        szp_float_decompress_single_thread_arg_profile(decompressed_data, num_elements, abs_err_bound, algo_block_size, compressed_data);
        
        // New buffer versions
        szp_float_single_thread_arg_buffer_profile(compressed_data, original_data, &compressed_size, abs_err_bound, num_elements, algo_block_size);
        szp_float_decompress_single_thread_arg_buffer_profile(decompressed_data, num_elements, abs_err_bound, algo_block_size, compressed_data);

#ifdef _OPENMP
        // Original versions
        szp_float_openmp_threadblock_arg_profile(compressed_data, original_data, &compressed_size, abs_err_bound, num_elements, algo_block_size);
        szp_float_decompress_openmp_threadblock_arg_profile(decompressed_data, num_elements, abs_err_bound, algo_block_size, compressed_data);
        
        // New buffer versions
        szp_float_openmp_threadblock_arg_buffer_profile(compressed_data, original_data, &compressed_size, abs_err_bound, num_elements, algo_block_size);
        szp_float_decompress_openmp_threadblock_arg_buffer_profile(decompressed_data, num_elements, abs_err_bound, algo_block_size, compressed_data);
#endif
    }
    printf("Warmup complete.\n");

    // --- Measurement Phase ---
    printf("\n--- Running %d Measurement Repetitions ---\n", repetitions);
    std::vector<struct szp_prof_data> st_comp_results, st_decomp_results, omp_comp_results, omp_decomp_results;
    std::vector<struct szp_prof_data> st_comp_buffer_results, st_decomp_buffer_results, omp_comp_buffer_results, omp_decomp_buffer_results;

    for (int i = 0; i < repetitions; ++i) {
        // === Original Arg Versions ===
        st_comp_results.push_back(szp_float_single_thread_arg_profile(compressed_data, original_data, &compressed_size, abs_err_bound, num_elements, algo_block_size));
        st_decomp_results.push_back(szp_float_decompress_single_thread_arg_profile(decompressed_data, num_elements, abs_err_bound, algo_block_size, compressed_data));

        // === New Buffer Versions ===
        st_comp_buffer_results.push_back(szp_float_single_thread_arg_buffer_profile(compressed_data, original_data, &compressed_size, abs_err_bound, num_elements, algo_block_size));
        st_decomp_buffer_results.push_back(szp_float_decompress_single_thread_arg_buffer_profile(decompressed_data, num_elements, abs_err_bound, algo_block_size, compressed_data));

        if (i == 0) { // Verify correctness on first run
            double max_abs_err = 0.0;
            for (size_t j = 0; j < num_elements; ++j) {
                max_abs_err = std::max(max_abs_err, (double)fabs(original_data[j] - decompressed_data[j]));
            }
            printf("\n--- Intermediate Verification (Single-Thread Rep %d) ---\n", i + 1);
            printf("  - Max Abs Error: %.6e (Bound: %.2e)\n", max_abs_err, abs_err_bound);
        }

#ifdef _OPENMP
        // === Original Arg Versions ===
        omp_comp_results.push_back(szp_float_openmp_threadblock_arg_profile(compressed_data, original_data, &compressed_size, abs_err_bound, num_elements, algo_block_size));
        omp_decomp_results.push_back(szp_float_decompress_openmp_threadblock_arg_profile(decompressed_data, num_elements, abs_err_bound, algo_block_size, compressed_data));
        
        // === New Buffer Versions ===
        omp_comp_buffer_results.push_back(szp_float_openmp_threadblock_arg_buffer_profile(compressed_data, original_data, &compressed_size, abs_err_bound, num_elements, algo_block_size));
        omp_decomp_buffer_results.push_back(szp_float_decompress_openmp_threadblock_arg_buffer_profile(decompressed_data, num_elements, abs_err_bound, algo_block_size, compressed_data));

        if (i == 0) { // Verify correctness on first run
            double max_abs_err = 0.0;
            for (size_t j = 0; j < num_elements; ++j) {
                max_abs_err = std::max(max_abs_err, (double)fabs(original_data[j] - decompressed_data[j]));
            }
            printf("\n--- Intermediate Verification (OpenMP Rep %d) ---\n", i + 1);
            printf("  - Max Abs Error: %.6e (Bound: %.2e)\n", max_abs_err, abs_err_bound);
        }
#endif
    }
    printf("Measurements complete.\n");

    // --- Analysis and Reporting ---
    print_statistics("Single-Threaded Compression (Original Arg)", st_comp_results, num_elements, true, false);
    print_statistics("Single-Threaded Compression (Buffer Arg)", st_comp_buffer_results, num_elements, true, false);
    
    print_statistics("Single-Threaded Decompression (Original Arg)", st_decomp_results, num_elements, false, false);
    print_statistics("Single-Threaded Decompression (Buffer Arg)", st_decomp_buffer_results, num_elements, false, false);
#ifdef _OPENMP
    print_statistics("OpenMP Compression (Original Arg)", omp_comp_results, num_elements, true, true);
    print_statistics("OpenMP Compression (Buffer Arg)", omp_comp_buffer_results, num_elements, true, true);

    print_statistics("OpenMP Decompression (Original Arg)", omp_decomp_results, num_elements, false, true);
    print_statistics("OpenMP Decompression (Buffer Arg)", omp_decomp_buffer_results, num_elements, false, true);
#endif

    // Cleanup
    free(original_data);
    free(compressed_data);
    free(decompressed_data);

    printf("\n========================================================================================\n");
    printf("Profiling test complete.\n");
    return 0;
}
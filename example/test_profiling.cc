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

#ifdef _OPENMP
#include "omp.h"
#endif

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

struct MethodBenchResult {
    const char* name;
    size_t compressed_size;
    double raw_size_gb;
    double compression_ratio;
    double avg_compression_ratio;
    double median_compression_ratio;
    double p95_compression_ratio;
    double std_compression_ratio;
    double avg_comp_throughput_gbs;
    double median_comp_throughput_gbs;
    double p95_comp_throughput_gbs;
    double std_comp_throughput_gbs;
    double avg_decomp_throughput_gbs;
    double median_decomp_throughput_gbs;
    double p95_decomp_throughput_gbs;
    double std_decomp_throughput_gbs;
    double max_abs_err;
    double max_rel_err;
    double psnr;
    double avg_comp_time;
    double avg_decomp_time;
};

typedef void (*migrated_compress_fn)(unsigned char*, float*, size_t*, float, size_t, int);
typedef void (*migrated_decompress_fn)(float*, size_t, float, int, unsigned char*);

static double wall_time_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static double percentile_from_sorted(const std::vector<double>& sorted_values,
                                     double percentile) {
    if (sorted_values.empty()) {
        return 0.0;
    }
    if (sorted_values.size() == 1) {
        return sorted_values[0];
    }

    double clamped = std::max(0.0, std::min(1.0, percentile));
    double pos = clamped * (double)(sorted_values.size() - 1);
    size_t lo = (size_t)std::floor(pos);
    size_t hi = (size_t)std::ceil(pos);
    double frac = pos - (double)lo;
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
}

struct ScalarStats {
    double avg;
    double median;
    double p95;
    double stddev;
};

static ScalarStats compute_scalar_stats(const std::vector<double>& values) {
    ScalarStats stats = {};
    if (values.empty()) {
        return stats;
    }

    std::vector<double> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());

    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    stats.avg = sum / (double)values.size();
    stats.median = percentile_from_sorted(sorted_values, 0.5);
    stats.p95 = percentile_from_sorted(sorted_values, 0.95);

    double variance = 0.0;
    for (double value : values) {
        double delta = value - stats.avg;
        variance += delta * delta;
    }
    stats.stddev = std::sqrt(variance / (double)values.size());
    return stats;
}

static MethodBenchResult run_migrated_method(
    const char* name,
    migrated_compress_fn compress_fn,
    migrated_decompress_fn decompress_fn,
    float* original_data,
    size_t num_elements,
    float abs_err_bound,
    float value_range,
    int block_size,
    int warmup_runs,
    int repetitions,
    unsigned char* compressed_data,
    float* decompressed_data)
{
    MethodBenchResult result = {};
    result.name = name;
    const double raw_size_bytes = (double)(num_elements * sizeof(float));
    const double raw_size_gb = raw_size_bytes / (1024.0 * 1024.0 * 1024.0);
    result.raw_size_gb = raw_size_gb;

    for (int i = 0; i < warmup_runs; ++i) {
        size_t warmup_size = 0;
        compress_fn(compressed_data, original_data, &warmup_size, abs_err_bound,
                    num_elements, block_size);
        decompress_fn(decompressed_data, num_elements, abs_err_bound, block_size,
                      compressed_data + sizeof(float));
    }

    std::vector<double> comp_times;
    std::vector<double> decomp_times;
    std::vector<double> comp_throughputs;
    std::vector<double> decomp_throughputs;
    std::vector<double> comp_ratios;
    comp_times.reserve((size_t)repetitions);
    decomp_times.reserve((size_t)repetitions);
    comp_throughputs.reserve((size_t)repetitions);
    decomp_throughputs.reserve((size_t)repetitions);
    comp_ratios.reserve((size_t)repetitions);

    double psnr_sum = 0.0;
    result.max_abs_err = 0.0;
    result.max_rel_err = 0.0;

    for (int i = 0; i < repetitions; ++i) {
        double t0 = wall_time_seconds();
        compress_fn(compressed_data, original_data, &result.compressed_size,
                    abs_err_bound, num_elements, block_size);
        double t1 = wall_time_seconds();
        decompress_fn(decompressed_data, num_elements, abs_err_bound, block_size,
                      compressed_data + sizeof(float));
        double t2 = wall_time_seconds();

        double comp_time = t1 - t0;
        double decomp_time = t2 - t1;
        double comp_ratio =
            (result.compressed_size == 0)
                ? 0.0
                : raw_size_bytes / (double)result.compressed_size;

        comp_times.push_back(comp_time);
        decomp_times.push_back(decomp_time);
        comp_throughputs.push_back(
            (comp_time == 0.0) ? 0.0 : raw_size_gb / comp_time);
        decomp_throughputs.push_back(
            (decomp_time == 0.0) ? 0.0 : raw_size_gb / decomp_time);
        comp_ratios.push_back(comp_ratio);

        double mse_sum = 0.0;
        double diff_max = 0.0;
        for (size_t j = 0; j < num_elements; ++j) {
            double err =
                fabs((double)decompressed_data[j] - (double)original_data[j]);
            diff_max = std::max(diff_max, err);
            mse_sum += err * err;
        }
        double mse = (num_elements == 0) ? 0.0 : mse_sum / (double)num_elements;
        double max_rel_err =
            (value_range == 0.0f) ? 0.0 : diff_max / (double)value_range;
        double psnr =
            (mse == 0.0)
                ? 999.0
                : 20.0 * log10((double)value_range) - 10.0 * log10(mse);

        result.max_abs_err = std::max(result.max_abs_err, diff_max);
        result.max_rel_err = std::max(result.max_rel_err, max_rel_err);
        psnr_sum += psnr;
    }

    ScalarStats comp_time_stats = compute_scalar_stats(comp_times);
    ScalarStats decomp_time_stats = compute_scalar_stats(decomp_times);
    ScalarStats comp_thr_stats = compute_scalar_stats(comp_throughputs);
    ScalarStats decomp_thr_stats = compute_scalar_stats(decomp_throughputs);
    ScalarStats comp_ratio_stats = compute_scalar_stats(comp_ratios);

    result.avg_comp_time = comp_time_stats.avg;
    result.avg_decomp_time = decomp_time_stats.avg;
    result.compression_ratio =
        comp_ratios.empty() ? 0.0 : comp_ratios.back();
    result.avg_compression_ratio = comp_ratio_stats.avg;
    result.median_compression_ratio = comp_ratio_stats.median;
    result.p95_compression_ratio = comp_ratio_stats.p95;
    result.std_compression_ratio = comp_ratio_stats.stddev;
    result.avg_comp_throughput_gbs = comp_thr_stats.avg;
    result.median_comp_throughput_gbs = comp_thr_stats.median;
    result.p95_comp_throughput_gbs = comp_thr_stats.p95;
    result.std_comp_throughput_gbs = comp_thr_stats.stddev;
    result.avg_decomp_throughput_gbs = decomp_thr_stats.avg;
    result.median_decomp_throughput_gbs = decomp_thr_stats.median;
    result.p95_decomp_throughput_gbs = decomp_thr_stats.p95;
    result.std_decomp_throughput_gbs = decomp_thr_stats.stddev;
    result.psnr = psnr_sum / (double)repetitions;
    return result;
}

static int run_migrated_suite(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(stderr,
                "Usage: %s --migrated <data_file> <rel_err_bound> <block_size> "
                "[-w warmups] [-r reps]\n",
                argv[0]);
        return 1;
    }

    const char* data_file_path = argv[2];
    float rel_err_bound = (float)atof(argv[3]);
    int block_size = atoi(argv[4]);
    int warmup_runs = 1;
    int repetitions = 3;

    for (int i = 5; i < argc; ++i) {
        if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            warmup_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            repetitions = atoi(argv[++i]);
        }
    }

    int status = 0;
    size_t num_elements = 0;
    float* original_data =
        szp_readFloatData(const_cast<char*>(data_file_path), &num_elements,
                         &status);
    if (status != SZ_SCES || original_data == nullptr || num_elements == 0) {
        fprintf(stderr, "Error: cannot read float data from %s\n",
                data_file_path);
        free(original_data);
        return 1;
    }

    float vmin = original_data[0];
    float vmax = original_data[0];
    for (size_t i = 1; i < num_elements; ++i) {
        vmin = std::min(vmin, original_data[i]);
        vmax = std::max(vmax, original_data[i]);
    }
    float value_range = vmax - vmin;
    if (value_range == 0.0f) {
        value_range = 1.0f;
    }
    float abs_err_bound = rel_err_bound * value_range;

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    size_t max_compressed_size = sizeof(float) +
                                 (size_t)max_threads * sizeof(size_t) +
                                 num_elements * (sizeof(float) + 1U) + 4096U;

    std::vector<unsigned char> compressed_data(max_compressed_size);
    std::vector<float> decompressed_data(num_elements);

    struct MethodEntry {
        const char* name;
        migrated_compress_fn compress_fn;
        migrated_decompress_fn decompress_fn;
    };

    const MethodEntry methods[] = {
        {"BlockAligned", szp_float_compress_blockaligned,
         szp_float_decompress_blockaligned},
        {"VecBA-3pass", szp_float_compress_vecBlockaligned,
         szp_float_decompress_vecBlockaligned},
        {"VecBA-SinglePass", szp_float_compress_vecBlockaligned_singlepass,
         szp_float_decompress_vecBlockaligned_singlepass},
    };

    std::vector<MethodBenchResult> results;
    results.reserve(sizeof(methods) / sizeof(methods[0]));

    printf("\n--- Migrated ZCCL Method Benchmark ---\n");
    printf("Data File: %s\n", data_file_path);
    printf("Elements: %s\n", format_with_commas(num_elements).c_str());
    printf("Range: [%.6g, %.6g], value_range=%.6g\n", vmin, vmax, value_range);
    printf("Rel Err Bound: %.2e\n", rel_err_bound);
    printf("Abs Err Bound: %.6e\n", abs_err_bound);
    printf("Block Size: %d\n", block_size);
    printf("Warmup Runs: %d\n", warmup_runs);
    printf("Repetitions: %d\n", repetitions);
    printf("OpenMP Threads: %d\n", max_threads);
    printf("========================================================================================\n");

    for (const auto& method : methods) {
        MethodBenchResult result = run_migrated_method(
            method.name, method.compress_fn, method.decompress_fn, original_data,
            num_elements, abs_err_bound, value_range, block_size, warmup_runs,
            repetitions, compressed_data.data(), decompressed_data.data());
        results.push_back(result);
    }

    printf("%-20s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
           "Method", "CmpSize", "CR(avg)", "CmpGB/s",
           "DecGB/s", "MaxAbsErr", "MaxRelErr", "PSNR");
    for (const auto& result : results) {
        printf("%-20s %-12zu %-12.6f %-12.6f %-12.6f %-12.6e %-12.6e %-12.6f\n",
               result.name, result.compressed_size,
               result.avg_compression_ratio, result.avg_comp_throughput_gbs,
               result.avg_decomp_throughput_gbs, result.max_abs_err,
               result.max_rel_err, result.psnr);
    }

    printf("\n--- Detailed Method Metrics ---\n");
    for (const auto& result : results) {
        printf("%s\n", result.name);
        printf("  Compression throughput (GB/s): avg=%.6f median=%.6f p95=%.6f std=%.6f\n",
               result.avg_comp_throughput_gbs,
               result.median_comp_throughput_gbs,
               result.p95_comp_throughput_gbs,
               result.std_comp_throughput_gbs);
        printf("  Compression ratio (CR): avg=%.6f median=%.6f p95=%.6f std=%.6f size=%zu\n",
               result.avg_compression_ratio,
               result.median_compression_ratio,
               result.p95_compression_ratio,
               result.std_compression_ratio,
               result.compressed_size);
        printf("  Decompression throughput (GB/s): avg=%.6f median=%.6f p95=%.6f std=%.6f\n",
               result.avg_decomp_throughput_gbs,
               result.median_decomp_throughput_gbs,
               result.p95_decomp_throughput_gbs,
               result.std_decomp_throughput_gbs);
        printf("  Error metrics: maxAbsErr=%.9e maxRelErr=%.9e PSNR=%.9f\n",
               result.max_abs_err, result.max_rel_err, result.psnr);
    }

    bool consistent = true;
    if (!results.empty()) {
        const auto& baseline = results.front();
        const double err_tol =
            std::max(1e-8, (double)abs_err_bound * 1e-6);
        const double psnr_tol = 1e-3;
        const double cr_tol = 1e-9;

        printf("\n--- Pairwise Differences (vs %s) ---\n", baseline.name);
        for (size_t i = 1; i < results.size(); ++i) {
            const auto& result = results[i];
            double cr_diff =
                fabs(result.compression_ratio - baseline.compression_ratio);
            double err_diff = fabs(result.max_abs_err - baseline.max_abs_err);
            double psnr_diff = fabs(result.psnr - baseline.psnr);
            bool same = (result.compressed_size == baseline.compressed_size) &&
                        cr_diff <= cr_tol && err_diff <= err_tol &&
                        psnr_diff <= psnr_tol;

            consistent = consistent && same;
            printf("%-20s size_diff=%zd cr_diff=%.6e err_diff=%.6e psnr_diff=%.6e %s\n",
                   result.name,
                   (ptrdiff_t)result.compressed_size -
                       (ptrdiff_t)baseline.compressed_size,
                   cr_diff, err_diff, psnr_diff,
                   same ? "OK" : "DIFF");
        }
    }

    printf("\nMigration metric consistency: %s\n",
           consistent ? "PASS" : "FAIL");

    free(original_data);
    return consistent ? 0 : 2;
}

// --- Main Test Logic ---

void print_usage() {
    printf("Usage: ./test_profiling <data_file> <num_elements> <err_bound> <algo_block_size> [-w warmups] [-r reps]\n");
    printf("   or: ./test_profiling --migrated <data_file> <rel_err_bound> <block_size> [-w warmups] [-r reps]\n");
    printf("\n  <data_file>       : Path to the data file (.npy, .f32, .dat supported).\n");
    printf("  <num_elements>    : Number of elements to read from the data file.\n");
    printf("  <err_bound>       : Absolute error bound for compression (e.g., 1e-4).\n");
    printf("  <algo_block_size> : Internal block size for the SZP algorithm (e.g., 1024).\n");
    printf("  <rel_err_bound>   : Relative error bound for migrated-method comparison.\n");
    printf("  -w <warmups>      : Number of warmup runs (default: 10).\n");
    printf("  -r <reps>         : Number of repetitions for measurement (default: 30).\n");
    printf("\nExample:\n");
    printf("  ./test_profiling ../node_feat.f32 100000 1e-4 1024 -w 5 -r 20\n");
    printf("  ./test_profiling --migrated ../node_feat.f32 1e-4 128 -w 1 -r 3\n");
}

int main(int argc, char *argv[]) {
    if (argc >= 2 && strcmp(argv[1], "--migrated") == 0) {
        return run_migrated_suite(argc, argv);
    }

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

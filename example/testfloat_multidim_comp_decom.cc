/**
 *  @file test_multidim.cc
 *  @author Jiefeng Zhou
 *  @date Aug, 2025
 *  @brief A benchmark tool for compressing and decompressing multi-dimensional datasets.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "szp.h"
#include <sys/time.h>
#ifdef _OPENMP
#include "omp.h"
#endif
#include <vector>
#include <algorithm>
#include <numeric>

// --- Utility Functions ---

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

// --- Result Structures and Analysis ---

struct CompResult {
    double timeCost;
    size_t compressedSize;
    double compressionRatio;
};

struct DecompResult {
    double timeCost;
    double maxAbsErr;
    double psnr;
    double nrmse;
    double compressionRatio;
};

typedef void (*FloatCompressMethod)(unsigned char *, float *, size_t *, float,
                                    size_t, int);
typedef void (*FloatDecompressMethod)(float *, size_t, float, int,
                                      unsigned char *);

struct MethodComparisonResult {
    const char *name;
    double compTime;
    double decompTime;
    size_t compressedSize;
    double compressionRatio;
    double maxAbsErr;
    double psnr;
    double nrmse;
};


int calculateTrimmedStats(std::vector<double>& data, double& mean, double& stddev) {
    if (data.empty()) {
        mean = 0.0;
        stddev = 0.0;
        return 0;
    }

    std::sort(data.begin(), data.end());

    // Determine the number of outliers to trim (10% from top, 10% from bottom)
    int outliers_to_trim = static_cast<int>(data.size() * 0.1);
    int start_idx = outliers_to_trim;
    int end_idx = data.size() - 1 - outliers_to_trim;
    int valid_reps = end_idx - start_idx + 1;

    if (valid_reps <= 0) {
        // Not enough data to trim, fall back to calculating over the whole set or return 0
        mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
        stddev = (data.size() > 1) ? sqrt((sq_sum - data.size() * mean * mean) / (data.size() - 1)) : 0.0;
        return 0; // Indicate that trimming was not performed
    }

    double sum = 0.0;
    for (int i = start_idx; i <= end_idx; ++i) {
        sum += data[i];
    }
    mean = sum / valid_reps;

    double sum_sq_diff = 0.0;
    for (int i = start_idx; i <= end_idx; ++i) {
        sum_sq_diff += (data[i] - mean) * (data[i] - mean);
    }
    stddev = (valid_reps > 1) ? sqrt(sum_sq_diff / (valid_reps - 1)) : 0.0;

    return valid_reps;
}


// --- Compression Statistics ---
void writeCompToCsv(const char* filename, const std::vector<CompResult>& results) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    fprintf(fp, "Run,TimeSeconds,CompressedSize,CompressionRatio\n");
    for (size_t i = 0; i < results.size(); i++) {
        fprintf(fp, "%zu,%.6f,%zu,%.2f\n", i + 1, results[i].timeCost, results[i].compressedSize, results[i].compressionRatio);
    }
    fclose(fp);
}

void calculateCompStats(const std::vector<CompResult>& results) {
    if (results.empty()) return;
    std::vector<double> times, crs;
    for(const auto& res : results) {
        times.push_back(res.timeCost);
        crs.push_back(res.compressionRatio);
    }

    double timeAvg, timeStdDev, crAvg, crStdDev;
    int valid_reps = calculateTrimmedStats(times, timeAvg, timeStdDev);
    calculateTrimmedStats(crs, crAvg, crStdDev); // CRs are also sorted and trimmed

    int outliers_to_trim = static_cast<int>(results.size() * 0.1);

    printf("\n======= Compression Performance (over %zu runs) =======\n", results.size());
    if (valid_reps > 0) {
        printf("Note: Excluding %d slowest and %d fastest runs from statistics (%d/%zu runs used).\n",
               outliers_to_trim, outliers_to_trim, valid_reps, results.size());
    } else {
        printf("Warning: Not enough runs to trim outliers, using all %zu runs for statistics.\n", results.size());
    }

    
    printf("Time (s):   Avg=%.6f, StdDev=%.6f\n", timeAvg, timeStdDev);
    printf("Comp Ratio: Avg=%.2f, StdDev=%.2f\n", crAvg, crStdDev);
    printf("======================================================\n");
}


// --- Decompression Statistics ---
void writeDecompToCsv(const char* filename, const std::vector<DecompResult>& results) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    fprintf(fp, "Run,TimeSeconds,MaxAbsErr,PSNR,NRMSE,CompressionRatio\n");
    for (size_t i = 0; i < results.size(); i++) {
        fprintf(fp, "%zu,%.6f,%.6e,%.4f,%.6e,%.2f\n", i + 1, results[i].timeCost, results[i].maxAbsErr, results[i].psnr, results[i].nrmse, results[i].compressionRatio);
    }
    fclose(fp);
}

void calculateDecompStats(const std::vector<DecompResult>& results) {
    if (results.empty()) return;
    std::vector<double> times, psnrs;
    for(const auto& res : results) {
        times.push_back(res.timeCost);
        psnrs.push_back(res.psnr);
    }

    double timeAvg, timeStdDev, psnrAvg, psnrStdDev;
    int valid_reps = calculateTrimmedStats(times, timeAvg, timeStdDev);
    calculateTrimmedStats(psnrs, psnrAvg, psnrStdDev); // PSNRs are also sorted and trimmed

    int outliers_to_trim = static_cast<int>(results.size() * 0.1);
    
    printf("\n======= Decompression Performance (over %zu runs) =======\n", results.size());
    if (valid_reps > 0) {
        printf("Note: Excluding %d slowest and %d fastest runs from statistics (%d/%zu runs used).\n",
               outliers_to_trim, outliers_to_trim, valid_reps, results.size());
    } else {
        printf("Warning: Not enough runs to trim outliers, using all %zu runs for statistics.\n", results.size());
    }

    printf("Time (s): Avg=%.6f, StdDev=%.6f\n", timeAvg, timeStdDev);
    printf("PSNR:     Avg=%.4f, StdDev=%.4f\n", psnrAvg, psnrStdDev);
    printf("=========================================================\n");
}

double calculateAverage(const std::vector<double> &values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

void writeMethodComparisonCsv(const char *filename,
                              const std::vector<MethodComparisonResult> &results)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        return;
    }
    fprintf(fp,
            "Method,CompTimeSeconds,DecompTimeSeconds,CompressedSize,CompressionRatio,MaxAbsErr,PSNR,NRMSE\n");
    for (const auto &res : results) {
        fprintf(fp, "%s,%.6f,%.6f,%zu,%.8f,%.8e,%.8f,%.8e\n", res.name,
                res.compTime, res.decompTime, res.compressedSize,
                res.compressionRatio, res.maxAbsErr, res.psnr, res.nrmse);
    }
    fclose(fp);
}


template<typename T>
DecompResult evaluate(const T* ori_data, const T* dec_data, size_t num_elements, size_t compressed_size) {
    DecompResult res;
    double max_val = ori_data[0], min_val = ori_data[0], max_abs_err = 0.0, mse = 0.0;

    for (size_t i = 0; i < num_elements; ++i) {
        if (ori_data[i] > max_val) max_val = ori_data[i];
        if (ori_data[i] < min_val) min_val = ori_data[i];
        double abs_err = fabs((double)ori_data[i] - (double)dec_data[i]);
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        mse += abs_err * abs_err;
    }
    mse /= num_elements;
    double value_range = max_val - min_val;
    
    res.maxAbsErr = max_abs_err;
    res.psnr = (mse == 0) ? 999.99 : 20 * log10(value_range) - 10 * log10(mse);
    res.nrmse = (value_range == 0) ? 0 : sqrt(mse) / value_range;
    res.compressionRatio = (double)(num_elements * sizeof(T)) / compressed_size;

    printf("  Max Abs Error: %.6e, PSNR: %.4f, NRMSE: %.6e, CR: %.2f\n", res.maxAbsErr, res.psnr, res.nrmse, res.compressionRatio);
    return res;
}

int run_method_comparison(const char *filepath, float err_bound, int block_size,
                          int repetitions, int warmupRuns,
                          const char *csvFilePath)
{
    int status = 0;
    size_t num_elements = 0;
    float *data = szp_readFloatData((char *)filepath, &num_elements, &status);
    if (status != SZ_SCES || data == NULL) {
        fprintf(stderr, "Error reading %s\n", filepath);
        return 1;
    }

    const struct {
        const char *name;
        FloatCompressMethod compress;
        FloatDecompressMethod decompress;
    } methods[] = {
        {"blockaligned", szp_float_compress_blockaligned,
         szp_float_decompress_blockaligned},
        {"vecBlockaligned", szp_float_compress_vecBlockaligned,
         szp_float_decompress_vecBlockaligned},
        {"vecBlockaligned_singlepass",
         szp_float_compress_vecBlockaligned_singlepass,
         szp_float_decompress_vecBlockaligned_singlepass},
    };

    const size_t method_count = sizeof(methods) / sizeof(methods[0]);
#ifdef _OPENMP
    const size_t max_threads = (size_t)omp_get_max_threads();
#else
    const size_t max_threads = 1;
#endif
    const size_t max_compressed_size =
        sizeof(float) + max_threads * sizeof(size_t) +
        num_elements * (sizeof(float) + 1U) + 1024U;

    printf("--- Comparing Migrated ZCCL Compressor Methods ---\n");
    printf("Input file: %s\n", filepath);
    printf("Elements: %zu\n", num_elements);
    printf("Error bound: %.6e\n", err_bound);
    printf("Block size: %d\n", block_size);
    printf("Warmups: %d, repetitions: %d\n\n", warmupRuns, repetitions);

    std::vector<MethodComparisonResult> comparison_results;
    comparison_results.reserve(method_count);

    for (size_t method_idx = 0; method_idx < method_count; method_idx++) {
        unsigned char *compressed_data =
            (unsigned char *)malloc(max_compressed_size);
        float *decompressed_data =
            (float *)malloc(num_elements * sizeof(float));
        std::vector<double> comp_times;
        std::vector<double> decomp_times;
        size_t compressed_size = 0;

        if (compressed_data == NULL || decompressed_data == NULL) {
            fprintf(stderr, "Error: memory allocation failed for %s\n",
                    methods[method_idx].name);
            free(compressed_data);
            free(decompressed_data);
            free(data);
            return 1;
        }

        for (int run = 0; run < warmupRuns + repetitions; run++) {
            cost_start();
            methods[method_idx].compress(compressed_data, data, &compressed_size,
                                         err_bound, num_elements, block_size);
            cost_end();
            if (run >= warmupRuns) {
                comp_times.push_back(totalCost);
            }
        }

        for (int run = 0; run < warmupRuns + repetitions; run++) {
            cost_start();
            methods[method_idx].decompress(decompressed_data, num_elements,
                                           err_bound, block_size,
                                           compressed_data + sizeof(float));
            cost_end();
            if (run >= warmupRuns) {
                decomp_times.push_back(totalCost);
            }
        }

        DecompResult quality =
            evaluate<float>(data, decompressed_data, num_elements, compressed_size);
        MethodComparisonResult result = {
            methods[method_idx].name,
            calculateAverage(comp_times),
            calculateAverage(decomp_times),
            compressed_size,
            quality.compressionRatio,
            quality.maxAbsErr,
            quality.psnr,
            quality.nrmse,
        };
        comparison_results.push_back(result);

        free(compressed_data);
        free(decompressed_data);
    }

    printf("\n%-26s %-12s %-12s %-14s %-14s %-14s %-12s\n", "Method",
           "Comp(s)", "Decomp(s)", "CompSize", "CR", "MaxAbsErr", "PSNR");
    printf("%-26s %-12s %-12s %-14s %-14s %-14s %-12s\n",
           "--------------------------", "------------", "------------",
           "--------------", "--------------", "--------------",
           "------------");
    for (const auto &res : comparison_results) {
        printf("%-26s %-12.6f %-12.6f %-14zu %-14.8f %-14.8e %-12.6f\n",
               res.name, res.compTime, res.decompTime, res.compressedSize,
               res.compressionRatio, res.maxAbsErr, res.psnr);
    }

    if (csvFilePath != NULL && csvFilePath[0] != '\0') {
        writeMethodComparisonCsv(csvFilePath, comparison_results);
    }

    const MethodComparisonResult &reference = comparison_results.front();
    const double max_abs_tol = 1e-8;
    const double psnr_tol = 1e-6;
    bool consistent = true;
    for (size_t i = 0; i < comparison_results.size(); i++) {
        const MethodComparisonResult &res = comparison_results[i];
        if (res.compressedSize != reference.compressedSize ||
            fabs(res.maxAbsErr - reference.maxAbsErr) > max_abs_tol ||
            fabs(res.psnr - reference.psnr) > psnr_tol) {
            consistent = false;
        }
    }

    if (!consistent) {
        fprintf(stderr,
                "\nWarning: migrated methods show meaningful metric differences. This indicates the migration may still be incorrect.\n");
        free(data);
        return 2;
    }

    printf(
        "\nAll migrated methods are numerically consistent: Compression ratio, maxAbsErr, and PSNR show no meaningful differences.\n");
    free(data);
    return 0;
}

// --- Main Logic ---

void print_usage() {
    printf("Usage: test_multidim <mode> <options>\n");
    printf("\n  Mode: -c (compress) or -d (decompress)\n");
    printf("        -m (compare migrated ZCCL-derived float compressor methods)\n");
    printf("\n  Compress Mode:\n");
    printf("    ./test_multidim -c <filepath> <dtype> <err_bound> <block_size> <dims...> [-r reps] [-w warmups] [-o out.csv]\n");
    printf("    Example: ./test_multidim -c temp.f32 float 1e-4 128 512 512 512 -r 10 -w 2\n");
    printf("\n  Decompress Mode:\n");
    printf("    ./test_multidim -d <compressed_file> <dtype> <block_size> <dims...> [-r reps] [-w warmups] [-o out.csv]\n");
    printf("    Example: ./test_multidim -d temp.f32.szp float 128 512 512 512\n");
    printf("\n  Compare Mode:\n");
    printf("    ./test_multidim -m <filepath> <err_bound> <block_size> [-r reps] [-w warmups] [-o out.csv]\n");
    printf("    Example: ./test_multidim -m /mnt/c/.../aramco-snapshot-1420.f32 1e-4 1024 -r 3 -w 1\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    bool compress_mode = (strcmp(argv[1], "-c") == 0);
    bool decompress_mode = (strcmp(argv[1], "-d") == 0);
    bool compare_mode = (strcmp(argv[1], "-m") == 0);

    if (!compress_mode && !decompress_mode && !compare_mode) {
        print_usage();
        return 1;
    }

    if (compare_mode) {
        if (argc < 5) {
            print_usage();
            return 1;
        }

        const char *filepath = argv[2];
        float err_bound = (float)atof(argv[3]);
        int block_size = atoi(argv[4]);
        int repetitions = 1;
        int warmupRuns = 0;
        char csvFilePath[256] = {0};

        for (int i = 5; i < argc; i++) {
            if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
                repetitions = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
                warmupRuns = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
                snprintf(csvFilePath, sizeof(csvFilePath), "%s", argv[++i]);
            }
        }

        return run_method_comparison(filepath, err_bound, block_size,
                                     repetitions, warmupRuns, csvFilePath);
    }

    // --- Argument Parsing ---
    const char* filepath = argv[2];
    const char* datatype_str = argv[3];
    double err_bound = compress_mode ? atof(argv[4]) : 0;
    int block_size = compress_mode ? atoi(argv[5]) : atoi(argv[4]);
    
    int dim_start_index = compress_mode ? 6 : 5;
    size_t num_elements = 1;
    int current_arg = dim_start_index;
    while (current_arg < argc && argv[current_arg][0] != '-') {
        num_elements *= (size_t)atol(argv[current_arg]);
        current_arg++;
    }

    int repetitions = 1, warmupRuns = 0;
    char csvFilePath[256] = {0};
    while (current_arg < argc) {
        if (strcmp(argv[current_arg], "-r") == 0) repetitions = atoi(argv[++current_arg]);
        else if (strcmp(argv[current_arg], "-w") == 0) warmupRuns = atoi(argv[++current_arg]);
        else if (strcmp(argv[current_arg], "-o") == 0) sprintf(csvFilePath, "%s", argv[++current_arg]);
        current_arg++;
    }

    int sz_datatype = (strcmp(datatype_str, "float") == 0) ? SZ_FLOAT : SZ_DOUBLE;
    size_t dtype_size = (sz_datatype == SZ_FLOAT) ? sizeof(float) : sizeof(double);

    // --- Compression Logic ---
    if (compress_mode) {
        int status;
        void* data = (sz_datatype == SZ_FLOAT) ? (void*)szp_readFloatData((char*)filepath, &num_elements, &status) : (void*)szp_readDoubleData((char*)filepath, &num_elements, &status);
        if (status != SZ_SCES) {
            fprintf(stderr, "Error reading %s\n", filepath);
            return 1;
        }

        printf("--- Compressing %s ---\n", filepath);
        std::vector<CompResult> results;
        for (int i = 0; i < warmupRuns + repetitions; ++i) {
            size_t compressed_size;
            cost_start();
            unsigned char* bytes = szp_compress(SZP_RANDOMACCESS, sz_datatype, data, &compressed_size, ABS, err_bound, 0, num_elements, block_size);
            cost_end();

            if (i < warmupRuns) {
                printf("Warmup %d/%d: Time=%.4fs\n", i + 1, warmupRuns, totalCost);
            } else {
                printf("Run %d/%d: Time=%.4fs, Size=%zu, CR=%.2f\n", i - warmupRuns + 1, repetitions, totalCost, compressed_size, (double)(num_elements * dtype_size) / compressed_size);
                CompResult res = {totalCost, compressed_size, (double)(num_elements * dtype_size) / compressed_size};
                results.push_back(res);
                if (i == warmupRuns + repetitions - 1) {
                    char outpath[256];
                    sprintf(outpath, "%s.szp", filepath);
                    szp_writeByteData(bytes, compressed_size, outpath, &status);
                    printf("Compressed data written to %s\n", outpath);
                }
            }
            free(bytes);
        }
        free(data);
        if (results.size() > 0) calculateCompStats(results);
        if (csvFilePath[0]) writeCompToCsv(csvFilePath, results);
    }

    // --- Decompression Logic ---
    if (decompress_mode) {
        int status;
        size_t compressed_size;
        unsigned char* bytes = szp_readByteData((char*)filepath, &compressed_size, &status);
        if (status != SZ_SCES) {
            fprintf(stderr, "Error reading %s\n", filepath);
            return 1;
        }

        char oriFilePath[256];
        strcpy(oriFilePath, filepath);
        oriFilePath[strlen(filepath) - 4] = '\0'; // Remove .szp
        void* ori_data = (sz_datatype == SZ_FLOAT) ? (void*)szp_readFloatData(oriFilePath, &num_elements, &status) : (void*)szp_readDoubleData(oriFilePath, &num_elements, &status);
        if (status != SZ_SCES) {
            fprintf(stderr, "Error reading original file %s for verification\n", oriFilePath);
            free(bytes);
            return 1;
        }

        printf("--- Decompressing %s ---\n", filepath);
        std::vector<DecompResult> results;
        for (int i = 0; i < warmupRuns + repetitions; ++i) {
            cost_start();
            void* dec_data = szp_decompress(SZP_RANDOMACCESS, sz_datatype, bytes, compressed_size, num_elements, block_size);
            cost_end();

            if (i < warmupRuns) {
                printf("Warmup %d/%d: Time=%.4fs\n", i + 1, warmupRuns, totalCost);
            } else {
                printf("Run %d/%d: Time=%.4fs\n", i - warmupRuns + 1, repetitions, totalCost);
                DecompResult res = (sz_datatype == SZ_FLOAT) ? evaluate((float*)ori_data, (float*)dec_data, num_elements, compressed_size) : evaluate((double*)ori_data, (double*)dec_data, num_elements, compressed_size);
                res.timeCost = totalCost;
                results.push_back(res);
                if (i == warmupRuns + repetitions - 1) {
                    char outpath[256];
                    sprintf(outpath, "%s.out", filepath);
                    if(sz_datatype == SZ_FLOAT) szp_writeFloatData_inBytes((float*)dec_data, num_elements, outpath, &status);
                    else szp_writeDoubleData_inBytes((double*)dec_data, num_elements, outpath, &status);
                    printf("Decompressed data written to %s\n", outpath);
                }
            }
            free(dec_data);
        }
        free(bytes);
        free(ori_data);
        if (results.size() > 0) calculateDecompStats(results);
        if (csvFilePath[0]) writeDecompToCsv(csvFilePath, results);
    }

    return 0;
}

/**
 *  @file szp_profiling.h
 *  @author Jiefeng Zhou
 *  @date Oct, 2025
 */

#ifndef SZP_PROFILING_H
#define SZP_PROFILING_H

#ifdef __cplusplus
extern "C" {
#endif

struct szp_prof_data {
    double total;
    double allocation;
    double t_quant;
    double t_pred;
    double t_sign_abs_max;
    double t_packing;
    double t_unpacking;
    double t_sign_restore;
    double t_pred_recon;
    double t_dequant;
    size_t ops_packing;
    size_t ops_unpacking;
};

// Function declarations for profiling
struct szp_prof_data szp_float_single_thread_arg_profile(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                 size_t nbEle, int blockSize);

struct szp_prof_data szp_float_decompress_single_thread_arg_profile(float *newData, size_t nbEle,
                                            float absErrBound, int blockSize,
                                            unsigned char *cmpBytes);

struct szp_prof_data szp_float_openmp_threadblock_arg_profile(unsigned char *output, float *oriData, size_t *outSize, float absErrBound, size_t nbEle, int blockSize);

struct szp_prof_data szp_float_decompress_openmp_threadblock_arg_profile(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);


static size_t szp_float_block_compiler_buffer_profile(
    unsigned char *__restrict__ block_pointer, float *__restrict__ op,
    double inver_bound, size_t current_block_size, int *__restrict__ prior,
    unsigned char *__restrict__ temp_sign_arr,
    unsigned int *__restrict__ temp_predict_arr,
    int *__restrict__ temp_quant_arr,
    struct szp_prof_data* prof_data);

struct szp_prof_data szp_float_single_thread_arg_buffer_profile(
    unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize);
    
struct szp_prof_data szp_float_decompress_single_thread_arg_buffer_profile(
    float *newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes);

static size_t szp_float_decompress_block_compiler_buffer_profile(
    unsigned char *__restrict__ block_pointer,
    float *__restrict__ newData_perthread, size_t current_block_size,
    float absErrBound, int *__restrict__ prior,
    unsigned char *__restrict__ temp_sign_arr,
    unsigned int *__restrict__ temp_predict_arr,
    struct szp_prof_data* prof_data);

struct szp_prof_data szp_float_openmp_threadblock_arg_buffer_profile(
    unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize);

struct szp_prof_data szp_float_decompress_openmp_threadblock_arg_buffer_profile(
    float *newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes);

#ifdef __cplusplus
}
#endif

#endif /* SZP_PROFILING_H */
/**
 *  @file szp_profiling.h
 *  @author Jiefeng Zhou
 *  @date Oct, 2025
 */

#ifndef SZP_PROFILING_H
#define SZP_PROFILING_H

#include <stdio.h>
#include "szp.h"

struct szp_prof_data {
    double total;
    double allocation;

    // Timings
    double t_quant;
    double t_pred;
    double t_sign_abs_max;
    double t_packing;
    double t_unpacking;
    double t_sign_restore;
    double t_pred_recon;
    double t_dequant;

    // Operation Counts (Memory R/W)
    size_t ops_packing;
    size_t ops_unpacking;
};

#ifdef __cplusplus
extern "C" {
#endif

struct szp_prof_data szp_float_openmp_threadblock_arg_profile(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                      size_t nbEle, int blockSize);

struct szp_prof_data szp_float_single_thread_arg_profile(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                 size_t nbEle, int blockSize);

struct szp_prof_data szp_float_decompress_single_thread_arg_profile(float *newData, size_t nbEle,
                                            float absErrBound, int blockSize,
                                            unsigned char *cmpBytes);

struct szp_prof_data szp_float_decompress_openmp_threadblock_arg_profile(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);


#ifdef __cplusplus
}
#endif

#endif // SZP_PROFILING_H
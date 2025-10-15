/**
 *  @file szpd_Float.h
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#ifndef _szpd_Float_H
#define _szpd_Float_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include "szp_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

float *szp_float_decompress_openmp_threadblock(size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

void szp_float_decompress_openmp_threadblock_arg(float *newData, size_t nbEle, float absErrBound, int blockSize,
                                                 unsigned char *cmpBytes);


void szp_float_decompress_single_thread_arg_buffer(
float *__restrict__ newData, size_t nbEle, float absErrBound,
    const unsigned char *__restrict__ cmpBytes,
    unsigned char *__restrict__ temp_sign_arr,
    unsigned int *__restrict__ temp_predict_arr);

static size_t szp_float_decompress_block_compiler_buffer(
    const unsigned char *__restrict__ block_pointer,
    float *__restrict__ newData_perthread, size_t current_block_size,
    float absErrBound, int *__restrict__ prior,
    unsigned char *__restrict__ temp_sign_arr,
    unsigned int *__restrict__ temp_predict_arr);

void szp_float_decompress_openmp_threadblock_arg_buffer(
    float *newData, size_t nbEle, float absErrBound, int blockSize,
    const unsigned char *cmpBytes);

void szp_float_decompress_single_thread_arg(float *newData, size_t nbEle,
                                            float absErrBound, int blockSize, unsigned char *cmpBytes);

size_t szp_float_decompress_single_thread_arg_record(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

float *szp_float_decompress_openmp_threadblock_randomaccess(size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

void szp_float_decompress_openmp_threadblock_randomaccess_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _szpd_Float_H  ----- */

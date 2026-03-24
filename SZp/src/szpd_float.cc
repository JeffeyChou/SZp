/**
 *  @file szpd_float.h
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>, Sheng Di <sdi1@anl.gov>
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include "szpd_float.h"
#include <assert.h>
#include <math.h>
#include "szp_TypeManager.h"
#include "szp_CompressionToolkit.h"
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include "omp.h"
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

using namespace szp;

float *szp_float_decompress_openmp_threadblock(size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP
    float *newData = (float *)malloc(sizeof(float) * nbEle);
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;
    
    size_t threadblocksize = 0;
    int block_size = blockSize;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            threadblocksize = nbEle / nbThreads;
        }
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle; // Ensure the last thread processes all remaining elements
        }

        float *newData_perthread = newData + lo;
        size_t i = 0;
        size_t j = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int bit_count = 0;
        unsigned char *block_pointer = rcp + offsets[tid];

        float ori_prior = 0.0;
        float ori_current = 0.0;

        if (lo < hi) { // Ensure thread has data to process
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);

            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float)); 
            newData_perthread += 1;
        }
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo + 1; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            bit_count = block_pointer[0];
            block_pointer++;
            
            if (bit_count == 0)
            {
                ori_prior = (float)prior * absErrBound;
                
                for (j = 0; j < current_block_size; j++)
                {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                }
            }
            else
            {
                convertByteArray2IntArray_fast_1b_args(current_block_size, block_pointer, (current_block_size - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((current_block_size - 1) / 8 + 1);

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;
                for (j = 0; j < current_block_size; j++)
                {
                    if (temp_sign_arr[j] == 0)
                    {
                        diff = temp_predict_arr[j];
                    }
                    else
                    {
                        diff = 0 - temp_predict_arr[j];
                    }
                    current = prior + diff;
                    ori_current = (float)current * absErrBound;
                    prior = current;
                    memcpy(newData_perthread, &ori_current, sizeof(float));
                    newData_perthread++;
                }
            }
        }
        free(temp_predict_arr);
        free(temp_sign_arr);
    }
    return newData;

#else
    printf("Error! OpenMP not supported!\n");
    return NULL; 
#endif
}


void szp_float_decompress_single_thread_arg_buffer(
float *__restrict__ newData, size_t nbEle, float absErrBound,
    const unsigned char *__restrict__ cmpBytes,
    unsigned char *__restrict__ temp_sign_arr,
    unsigned int *__restrict__ temp_predict_arr)
{
    size_t *offsets = (size_t *)cmpBytes;

    const unsigned char *rcp = cmpBytes + sizeof(size_t);
    size_t block_size = nbEle;

    size_t lo = 0;
    size_t hi = nbEle;
    float *__restrict__ newData_perthread = newData;

    int prior = 0;

    unsigned int bit_count = 0;
    const unsigned char *block_pointer = rcp + offsets[0];
#ifdef SZP_DEBUG
    printf("Offset for thread 0: %zu\n", offsets[0]);
#endif // SZP_DEBUG

    float ori_prior = 0.0;

    if (lo < hi) { // Ensure there is at least one element to decompress
        memcpy(&prior, block_pointer, sizeof(int));
        block_pointer += sizeof(unsigned int);

        ori_prior = (float)prior * absErrBound;
        memcpy(newData_perthread, &ori_prior, sizeof(float));
        newData_perthread++;
    }

    for (size_t i = lo + 1; i < hi; i = i + block_size) {
        size_t current_block_size =
            (i + block_size > hi) ? (hi - i) : block_size;
        if (current_block_size == 0)
            continue;

        bit_count = block_pointer[0];
        block_pointer++;

#ifdef SZP_DEBUG
        printf("bit_count: %d\n", bit_count);
#endif // SZP_DEBUG

        if (bit_count == 0) {
            ori_prior = (float)prior * absErrBound;

            for (size_t j = 0; j < current_block_size; j++) {
                newData_perthread[j] = ori_prior;
            }
            newData_perthread += current_block_size;
        } else {
            convertByteArray2IntArray_fast_1b_args(
                current_block_size, const_cast<unsigned char *>(block_pointer),
                (current_block_size - 1) / 8 + 1, temp_sign_arr);
            block_pointer += ((current_block_size - 1) / 8 + 1);

            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                const_cast<unsigned char *>(block_pointer), current_block_size,
                temp_predict_arr, bit_count);
            block_pointer += savedbitsbytelength;

            // 1. Calculate the differences based on the sign array
            int *temp_int_predict_arr =
                reinterpret_cast<int *>(temp_predict_arr);
            for (size_t j = 0; j < current_block_size; j++) {
                if (temp_sign_arr[j] != 0) {
                    temp_int_predict_arr[j] =
                        -static_cast<int>(temp_predict_arr[j]);
                }
            }

#ifdef SZP_DEBUG
            printf("Decompression: temp_int_predict_arr = [");
            for (size_t j = 0; j < current_block_size; j++) {
                printf("%d", temp_int_predict_arr[j]);
                if (j < current_block_size - 1)
                    printf(", ");
            }
            printf("]\n");
#endif // SZP_DEBUG

            // 2. Prefix sum
            temp_int_predict_arr[0] += prior;
            for (size_t j = 1; j < current_block_size; j++) {
                temp_int_predict_arr[j] += temp_int_predict_arr[j - 1];
            }
            prior = temp_int_predict_arr[current_block_size - 1];

#ifdef SZP_DEBUG
            printf("Decompression: after prefix sum, temp_int_predict_arr = [");
            for (size_t j = 0; j < current_block_size; j++) {
                printf("%d", temp_int_predict_arr[j]);
                if (j < current_block_size - 1)
                    printf(", ");
        }
        printf("]\n");
#endif // SZP_DEBUG

            // 3. Dequantize and store the results
            for (size_t j = 0; j < current_block_size; j++) {
                newData_perthread[j] =
                    static_cast<float>(temp_int_predict_arr[j]) * absErrBound;
            }
#ifdef SZP_DEBUG
        printf("Decompression: written %zu elements to "
               "newData_perthread\n",
               current_block_size);
        for (size_t j = 0; j < current_block_size; j++) {
            printf("%f ", newData_perthread[j]);
        }
        printf("\n");
#endif // SZP_DEBUG
            newData_perthread += current_block_size;
        }
    }
}

void szp_float_decompress_single_thread_arg(float *newData, size_t nbEle,
                                            float absErrBound, int blockSize,
                                            unsigned char *cmpBytes)
{
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 1;
    
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    size_t block_size = blockSize;

    size_t lo = 0;
    size_t hi = nbEle;
    float *newData_perthread = newData;
    size_t i = 0;
    size_t j = 0;

    int prior = 0;
    int current = 0;
    int diff = 0;

    unsigned int bit_count = 0;
    unsigned char *block_pointer = rcp + offsets[0];

    float ori_prior = 0.0;
    float ori_current = 0.0;

    if (lo < hi) { // Ensure there is at least one element to decompress
        memcpy(&prior, block_pointer, sizeof(int));
        block_pointer += sizeof(unsigned int);

        ori_prior = (float)prior * absErrBound;
        memcpy(newData_perthread, &ori_prior, sizeof(float)); 
        newData_perthread += 1;
    }
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int savedbitsbytelength = 0;
    
    for (i = lo + 1; i < hi; i = i + block_size)
    {
        size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
        if (current_block_size == 0) continue;

        bit_count = block_pointer[0];
        block_pointer++;
        
        if (bit_count == 0)
        {
            ori_prior = (float)prior * absErrBound;
            
            for (j = 0; j < current_block_size; j++)
            {
                memcpy(newData_perthread, &ori_prior, sizeof(float));
                newData_perthread++;
            }
        }
        else
        {
            convertByteArray2IntArray_fast_1b_args(current_block_size, block_pointer, (current_block_size - 1) / 8 + 1, temp_sign_arr);
            block_pointer += ((current_block_size - 1) / 8 + 1);

            savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size, temp_predict_arr, bit_count);
            block_pointer += savedbitsbytelength;
            for (j = 0; j < current_block_size; j++)
            {
                if (temp_sign_arr[j] == 0)
                {
                    diff = temp_predict_arr[j];
                }
                else
                {
                    diff = 0 - temp_predict_arr[j];
                }
                current = prior + diff;
                ori_current = (float)current * absErrBound;
                prior = current;
                memcpy(newData_perthread, &ori_current, sizeof(float));
                newData_perthread++;
            }
        }
    }
    
    free(temp_predict_arr);
    free(temp_sign_arr);
}

size_t szp_float_decompress_single_thread_arg_record(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
    size_t total_memaccess = 0;
    
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 1;
    
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    size_t block_size = blockSize;

    size_t lo = 0;
    size_t hi = nbEle;
    float *newData_perthread = newData;
    size_t i = 0;
    size_t j = 0;

    int prior = 0;
    int current = 0;
    int diff = 0;

    unsigned int bit_count = 0;
    unsigned char *block_pointer = rcp + offsets[0]; 
    total_memaccess += sizeof(size_t); // Reading from offsets array

    float ori_prior = 0.0;
    float ori_current = 0.0;

    if (lo < hi) { // Ensure there is at least one element to decompress
        memcpy(&prior, block_pointer, sizeof(int));
        total_memaccess += sizeof(int) * 2; // read from block_pointer and write to prior
        block_pointer += sizeof(unsigned int);

        ori_prior = (float)prior * absErrBound;
        memcpy(newData_perthread, &ori_prior, sizeof(float)); 
        total_memaccess += sizeof(float) * 2; // read from ori_prior and write to newData_perthread
        newData_perthread += 1;
    }
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int savedbitsbytelength = 0;

    // Unified loop for all remaining data blocks
    for (i = lo + 1; i < hi; i = i + block_size)
    {
        size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
        if (current_block_size == 0) continue;

        bit_count = block_pointer[0];
        total_memaccess += sizeof(unsigned char); // Reading bit_count
        block_pointer++;
        
        if (bit_count == 0)
        {
            ori_prior = (float)prior * absErrBound;
            for (j = 0; j < current_block_size; j++)
            {
                memcpy(newData_perthread, &ori_prior, sizeof(float));
                total_memaccess += sizeof(float) * 2; // read from ori_prior and write to newData_perthread
                newData_perthread++;
            }
        }
        else
        {
            size_t sign_byte_len = (current_block_size - 1) / 8 + 1;
            convertByteArray2IntArray_fast_1b_args(current_block_size, block_pointer, sign_byte_len, temp_sign_arr);
            total_memaccess += sizeof(unsigned char) * sign_byte_len;         // Reading from block_pointer
            total_memaccess += sizeof(unsigned char) * current_block_size;    // Writing to temp_sign_arr
            block_pointer += sign_byte_len;

            savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size, temp_predict_arr, bit_count);
            total_memaccess += sizeof(unsigned char) * savedbitsbytelength;   // Reading from block_pointer
            total_memaccess += sizeof(unsigned int) * current_block_size;     // Writing to temp_predict_arr
            block_pointer += savedbitsbytelength;

            for (j = 0; j < current_block_size; j++)
            {
                if (temp_sign_arr[j] == 0)
                {
                    diff = temp_predict_arr[j];
                }
                else
                {
                    diff = 0 - temp_predict_arr[j];
                }
                total_memaccess += sizeof(unsigned char); // Reading from temp_sign_arr
                total_memaccess += sizeof(unsigned int);  // Reading from temp_predict_arr

                current = prior + diff;
                ori_current = (float)current * absErrBound;
                prior = current;
                memcpy(newData_perthread, &ori_current, sizeof(float));
                total_memaccess += sizeof(float) * 2;     // read from ori_current and write to newData_perthread
                newData_perthread++;
            }
        }
    }
    
    free(temp_predict_arr);
    free(temp_sign_arr);
    return total_memaccess;
}

void szp_float_decompress_openmp_threadblock_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP
    
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;
    
    size_t threadblocksize = 0;
    size_t block_size = blockSize;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            threadblocksize = nbEle / nbThreads;
        }
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle; // Ensure the last thread processes all remaining elements
        }

        float *newData_perthread = newData + lo;
        size_t i = 0;
        size_t j = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int bit_count = 0;
        unsigned char *block_pointer = rcp + offsets[tid];

        float ori_prior = 0.0;
        float ori_current = 0.0;

        if (lo < hi) { // Ensure thread has data to process
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);

            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float)); 
            newData_perthread += 1;
        }
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo + 1; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            bit_count = block_pointer[0];
            block_pointer++;
            
            if (bit_count == 0)
            {
                ori_prior = (float)prior * absErrBound;
                
                for (j = 0; j < current_block_size; j++)
                {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                }
            }
            else
            {
                convertByteArray2IntArray_fast_1b_args(current_block_size, block_pointer, (current_block_size - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((current_block_size - 1) / 8 + 1);

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;
                for (j = 0; j < current_block_size; j++)
                {
                    if (temp_sign_arr[j] == 0)
                    {
                        diff = temp_predict_arr[j];
                    }
                    else
                    {
                        diff = 0 - temp_predict_arr[j];
                    }
                    current = prior + diff;
                    ori_current = (float)current * absErrBound;
                    prior = current;
                    memcpy(newData_perthread, &ori_current, sizeof(float));
                    newData_perthread++;
                }
            }
        }
        free(temp_predict_arr);
        free(temp_sign_arr);
    }

#else
    printf("Error! OpenMP not supported!\n");
#endif
}

static size_t szp_float_decompress_block_compiler_buffer(
    const unsigned char *__restrict__ block_pointer,
    float *__restrict__ newData_perthread, size_t current_block_size,
    float absErrBound, int *__restrict__ prior,
    unsigned char *__restrict__ temp_sign_arr,
    unsigned int *__restrict__ temp_predict_arr)
{
    const unsigned char *start_block_pointer = block_pointer;
    unsigned int bit_count = block_pointer[0];
    block_pointer++;

    if (bit_count == 0) {
        float ori_prior = (float)(*prior) * absErrBound;
        for (size_t j = 0; j < current_block_size; j++) {
            newData_perthread[j] = ori_prior;
        }
    } else {
        size_t sign_byte_len = (current_block_size - 1) / 8 + 1;
        convertByteArray2IntArray_fast_1b_args(
            current_block_size, const_cast<unsigned char *>(block_pointer),
            sign_byte_len, temp_sign_arr);
        block_pointer += sign_byte_len;

        unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(
            const_cast<unsigned char *>(block_pointer), current_block_size,
            temp_predict_arr, bit_count);
        block_pointer += savedbitsbytelength;

        int *temp_int_predict_arr = reinterpret_cast<int *>(temp_predict_arr);
        for (size_t j = 0; j < current_block_size; j++) {
            if (temp_sign_arr[j] != 0) {
                temp_int_predict_arr[j] =
                    -static_cast<int>(temp_predict_arr[j]);
            }
        }

        temp_int_predict_arr[0] += *prior;
        for (size_t j = 1; j < current_block_size; j++) {
            temp_int_predict_arr[j] += temp_int_predict_arr[j - 1];
        }
        *prior = temp_int_predict_arr[current_block_size - 1];

        for (size_t j = 0; j < current_block_size; j++) {
            newData_perthread[j] =
                static_cast<float>(temp_int_predict_arr[j]) * absErrBound;
        }
    }

    return (size_t)(block_pointer - start_block_pointer);
}

void szp_float_decompress_openmp_threadblock_arg_buffer(
    float *newData, size_t nbEle, float absErrBound, int blockSize,
    const unsigned char *cmpBytes)
{
#ifdef _OPENMP
    const size_t *offsets = (const size_t *)cmpBytes;
    const unsigned char *rcp = NULL;
    unsigned int nbThreads = 0;
    size_t threadblocksize = 0;
    size_t block_size = (size_t)blockSize;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            threadblocksize = (nbEle + nbThreads - 1) / nbThreads;
        }

        int tid = omp_get_thread_num();
        size_t lo = (size_t)tid * threadblocksize;
        size_t hi = (size_t)(tid + 1) * threadblocksize;
        if (hi > nbEle) {
            hi = nbEle;
        }

        if (lo < hi) {
            float *newData_perthread = newData + lo;
            const unsigned char *block_pointer = rcp + offsets[tid];
            int prior = 0;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            *newData_perthread = (float)prior * absErrBound;
            newData_perthread++;

            unsigned char *temp_sign_arr =
                (unsigned char *)malloc(block_size * sizeof(unsigned char));
            unsigned int *temp_predict_arr =
                (unsigned int *)malloc(block_size * sizeof(unsigned int));

            for (size_t i = lo + 1; i < hi; i += block_size) {
                size_t current_block_size =
                    (i + block_size > hi) ? (hi - i) : block_size;
                if (current_block_size == 0) {
                    continue;
                }

                size_t consumed_bytes = szp_float_decompress_block_compiler_buffer(
                    block_pointer, newData_perthread, current_block_size,
                    absErrBound, &prior, temp_sign_arr, temp_predict_arr);
                block_pointer += consumed_bytes;
                newData_perthread += current_block_size;
            }

            free(temp_predict_arr);
            free(temp_sign_arr);
        }
    }
#else
    printf("Error! OpenMP not supported!\n");
#endif
}

static inline size_t szp_float_blockaligned_threadblocksize_decode(
    size_t nbEle, size_t block_size, unsigned int nbThreads)
{
    if (block_size == 0 || nbThreads == 0) {
        return 0;
    }
    return (nbEle / block_size / nbThreads) * block_size;
}

static inline void szp_float_fill_constant_block(float *dst, size_t count,
                                                 float value)
{
    size_t j = 0;

#ifdef __AVX512F__
    __m512 vvalue512 = _mm512_set1_ps(value);
    for (; j + 16 <= count; j += 16) {
        _mm512_storeu_ps(dst + j, vvalue512);
    }
#elif defined(__AVX2__)
    __m256 vvalue256 = _mm256_set1_ps(value);
    for (; j + 8 <= count; j += 8) {
        _mm256_storeu_ps(dst + j, vvalue256);
    }
#endif

    for (; j < count; j++) {
        dst[j] = value;
    }
}

void szp_float_decompress_blockaligned(float *newData, size_t nbEle,
                                       float absErrBound, int blockSize,
                                       unsigned char *cmpBytes)
{
#ifdef _OPENMP
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp = NULL;
    size_t block_size = (size_t)blockSize;
    unsigned int nbThreads = 0;
    size_t threadblocksize = 0;

#pragma omp parallel
    {
        int tid = 0;
        size_t lo = 0;
        size_t hi = 0;
        float *newData_perthread = NULL;
        int prior = 0;
        unsigned char *block_pointer = NULL;
        unsigned int *temp_sign_arr = NULL;
        unsigned int *temp_predict_arr = NULL;

#pragma omp single
        {
            nbThreads = (unsigned int)omp_get_num_threads();
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            threadblocksize = szp_float_blockaligned_threadblocksize_decode(
                nbEle, block_size, nbThreads);
        }

        tid = omp_get_thread_num();
        lo = (size_t)tid * threadblocksize;
        hi = (size_t)(tid + 1) * threadblocksize;
        if (tid == (int)nbThreads - 1) {
            hi = nbEle;
        }

        newData_perthread = newData + lo;
        block_pointer = rcp + offsets[tid];

        if (lo < hi) {
            float ori_prior = 0.0f;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float));
            newData_perthread += 1;
        }

        temp_sign_arr =
            (unsigned int *)malloc(block_size * sizeof(unsigned int));
        temp_predict_arr =
            (unsigned int *)malloc(block_size * sizeof(unsigned int));

        for (size_t i = lo + 1; i < hi; i += block_size) {
            size_t current_block_size =
                (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) {
                continue;
            }

            unsigned int bit_count = (unsigned int)(*block_pointer++);
            if (bit_count == 0U) {
                float ori_prior = (float)prior * absErrBound;
                szp_float_fill_constant_block(newData_perthread,
                                              current_block_size, ori_prior);
                newData_perthread += current_block_size;
                continue;
            }

            size_t sign_bytes = (current_block_size + 7U) >> 3;
#ifdef __AVX2__
            Jiajun_convertByte2UInt_fast_1b_args_avx2(
                current_block_size, block_pointer, sign_bytes, temp_sign_arr);
#else
            Jiajun_convertByte2UInt_fast_1b_args(
                current_block_size, block_pointer, sign_bytes, temp_sign_arr);
#endif
            block_pointer += sign_bytes;

            size_t savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                block_pointer, current_block_size, temp_predict_arr, bit_count);
            block_pointer += savedbitsbytelength;

            for (size_t j = 0; j < current_block_size; j++) {
                if (temp_sign_arr[j] == 0U) {
                    prior += (int)temp_predict_arr[j];
                } else {
                    prior -= (int)temp_predict_arr[j];
                }
                *newData_perthread++ = (float)prior * absErrBound;
            }
        }

        free(temp_predict_arr);
        free(temp_sign_arr);
    }
#else
    printf("Error! OpenMP not supported!\n");
#endif
}

void szp_float_decompress_vecBlockaligned(float *newData, size_t nbEle,
                                          float absErrBound, int blockSize,
                                          unsigned char *cmpBytes)
{
#ifdef _OPENMP
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp = NULL;
    size_t block_size = (size_t)blockSize;
    unsigned int nbThreads = 0;
    size_t threadblocksize = 0;

#pragma omp parallel
    {
        int tid = 0;
        size_t lo = 0;
        size_t hi = 0;
        float *newData_perthread = NULL;
        int prior = 0;
        unsigned char *block_pointer = NULL;
        unsigned int *temp_sign_arr = NULL;
        unsigned int *temp_predict_arr = NULL;
        int *temp_diff_arr = NULL;
        int *temp_prefix_arr = NULL;

#pragma omp single
        {
            nbThreads = (unsigned int)omp_get_num_threads();
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            threadblocksize = szp_float_blockaligned_threadblocksize_decode(
                nbEle, block_size, nbThreads);
        }

        tid = omp_get_thread_num();
        lo = (size_t)tid * threadblocksize;
        hi = (size_t)(tid + 1) * threadblocksize;
        if (tid == (int)nbThreads - 1) {
            hi = nbEle;
        }

        newData_perthread = newData + lo;
        block_pointer = rcp + offsets[tid];

        if (lo < hi) {
            float ori_prior = 0.0f;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float));
            newData_perthread += 1;
        }

        temp_sign_arr =
            (unsigned int *)malloc(block_size * sizeof(unsigned int));
        temp_predict_arr =
            (unsigned int *)malloc(block_size * sizeof(unsigned int));
        temp_diff_arr = (int *)malloc(block_size * sizeof(int));
        temp_prefix_arr = (int *)malloc(block_size * sizeof(int));

        for (size_t i = lo + 1; i < hi; i += block_size) {
            size_t current_block_size =
                (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) {
                continue;
            }

            unsigned int bit_count = (unsigned int)(*block_pointer++);
            if (bit_count == 0U) {
                float ori_prior = (float)prior * absErrBound;
                szp_float_fill_constant_block(newData_perthread,
                                              current_block_size, ori_prior);
                newData_perthread += current_block_size;
                continue;
            }

            size_t sign_bytes = (current_block_size + 7U) >> 3;
#ifdef __AVX2__
            Jiajun_convertByte2UInt_fast_1b_args_avx2(
                current_block_size, block_pointer, sign_bytes, temp_sign_arr);
#else
            Jiajun_convertByte2UInt_fast_1b_args(
                current_block_size, block_pointer, sign_bytes, temp_sign_arr);
#endif
            block_pointer += sign_bytes;

            size_t savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                block_pointer, current_block_size, temp_predict_arr, bit_count);
            block_pointer += savedbitsbytelength;

#ifdef __AVX2__
            {
                __m256i vzero_i = _mm256_setzero_si256();
                size_t j = 0;

                for (; j + 8 <= current_block_size; j += 8) {
                    __m256i vmag = _mm256_loadu_si256(
                        (const __m256i *)(temp_predict_arr + j));
                    __m256i vneg = _mm256_sub_epi32(vzero_i, vmag);
                    __m256i vsign = _mm256_loadu_si256(
                        (const __m256i *)(temp_sign_arr + j));
                    __m256i vmask = _mm256_cmpgt_epi32(vsign, vzero_i);
                    __m256i vdiff = _mm256_blendv_epi8(vmag, vneg, vmask);
                    _mm256_storeu_si256((__m256i *)(temp_diff_arr + j), vdiff);
                }
                for (; j < current_block_size; j++) {
                    int d = (int)temp_predict_arr[j];
                    if (temp_sign_arr[j] != 0U) {
                        d = -d;
                    }
                    temp_diff_arr[j] = d;
                }
            }
#else
            for (size_t j = 0; j < current_block_size; j++) {
                int d = (int)temp_predict_arr[j];
                if (temp_sign_arr[j] != 0U) {
                    d = -d;
                }
                temp_diff_arr[j] = d;
            }
#endif

            {
                int running = prior;
                for (size_t j = 0; j < current_block_size; j++) {
                    running += temp_diff_arr[j];
                    temp_prefix_arr[j] = running;
                }
                prior = running;
            }

#ifdef __AVX2__
            {
                __m256 veb = _mm256_set1_ps(absErrBound);
                size_t j = 0;

                for (; j + 8 <= current_block_size; j += 8) {
                    __m256i vint = _mm256_loadu_si256(
                        (const __m256i *)(temp_prefix_arr + j));
                    __m256 vf = _mm256_cvtepi32_ps(vint);
                    vf = _mm256_mul_ps(vf, veb);
                    _mm256_storeu_ps(newData_perthread + j, vf);
                }
                for (; j < current_block_size; j++) {
                    newData_perthread[j] =
                        (float)temp_prefix_arr[j] * absErrBound;
                }
            }
#else
            for (size_t j = 0; j < current_block_size; j++) {
                newData_perthread[j] =
                    (float)temp_prefix_arr[j] * absErrBound;
            }
#endif
            newData_perthread += current_block_size;
        }

        free(temp_prefix_arr);
        free(temp_diff_arr);
        free(temp_predict_arr);
        free(temp_sign_arr);
    }
#else
    printf("Error! OpenMP not supported!\n");
#endif
}

static void szp_float_block_vecBlockaligned_singlepass_decode_scalar(
    float *newData, size_t current_block_size, float absErrBound, int *prior,
    const unsigned int *temp_sign_arr, const unsigned int *temp_predict_arr)
{
    int running = *prior;

    for (size_t j = 0; j < current_block_size; j++) {
        int diff = (int)temp_predict_arr[j];
        if (temp_sign_arr[j] != 0U) {
            diff = -diff;
        }
        running += diff;
        newData[j] = (float)running * absErrBound;
    }

    *prior = running;
}

#if defined(__AVX2__) || defined(__AVX512F__)
#ifdef __AVX2__
static inline __m256i szp_avx2_prefix_sum_epi32(__m256i values)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i idx1 = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    const __m256i idx2 = _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5);
    const __m256i idx4 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 2, 3);
    __m256i shifted = _mm256_permutevar8x32_epi32(values, idx1);

    shifted = _mm256_blend_epi32(shifted, zero, 0x01);
    values = _mm256_add_epi32(values, shifted);
    shifted = _mm256_permutevar8x32_epi32(values, idx2);
    shifted = _mm256_blend_epi32(shifted, zero, 0x03);
    values = _mm256_add_epi32(values, shifted);
    shifted = _mm256_permutevar8x32_epi32(values, idx4);
    shifted = _mm256_blend_epi32(shifted, zero, 0x0F);
    values = _mm256_add_epi32(values, shifted);
    return values;
}

static void szp_float_block_vecBlockaligned_singlepass_decode_SIMD(
    float *newData, size_t current_block_size, float absErrBound, int *prior,
    const unsigned int *temp_sign_arr, const unsigned int *temp_predict_arr)
{
    size_t j = 0;
    int running = *prior;
    const __m256i zero_i = _mm256_setzero_si256();
    const __m256 veb = _mm256_set1_ps(absErrBound);

    for (; j + 8 <= current_block_size; j += 8) {
        __m256i vmag =
            _mm256_loadu_si256((const __m256i *)(temp_predict_arr + j));
        __m256i vsign =
            _mm256_loadu_si256((const __m256i *)(temp_sign_arr + j));
        __m256i vneg = _mm256_sub_epi32(zero_i, vmag);
        __m256i vmask = _mm256_cmpgt_epi32(vsign, zero_i);
        __m256i vdiff = _mm256_blendv_epi8(vmag, vneg, vmask);
        __m256i vprefix = szp_avx2_prefix_sum_epi32(vdiff);
        __m256 vf;

        vprefix = _mm256_add_epi32(vprefix, _mm256_set1_epi32(running));
        vf = _mm256_mul_ps(_mm256_cvtepi32_ps(vprefix), veb);
        _mm256_storeu_ps(newData + j, vf);
        running = _mm256_extract_epi32(vprefix, 7);
    }

    for (; j < current_block_size; j++) {
        int diff = (int)temp_predict_arr[j];
        if (temp_sign_arr[j] != 0U) {
            diff = -diff;
        }
        running += diff;
        newData[j] = (float)running * absErrBound;
    }

    *prior = running;
}
#endif

#ifdef __AVX512F__
static inline __m512i szp_avx512_prefix_sum_epi32(__m512i values)
{
    const __m512i idx1 = _mm512_setr_epi32(
        0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    const __m512i idx2 = _mm512_setr_epi32(
        0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    const __m512i idx4 = _mm512_setr_epi32(
        0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    const __m512i idx8 = _mm512_setr_epi32(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7);

    values = _mm512_add_epi32(
        values, _mm512_maskz_permutexvar_epi32(0xFFFE, idx1, values));
    values = _mm512_add_epi32(
        values, _mm512_maskz_permutexvar_epi32(0xFFFC, idx2, values));
    values = _mm512_add_epi32(
        values, _mm512_maskz_permutexvar_epi32(0xFFF0, idx4, values));
    values = _mm512_add_epi32(
        values, _mm512_maskz_permutexvar_epi32(0xFF00, idx8, values));
    return values;
}

static void szp_float_block_vecBlockaligned_singlepass_decode_SIMD512(
    float *newData, size_t current_block_size, float absErrBound, int *prior,
    const unsigned int *temp_sign_arr, const unsigned int *temp_predict_arr)
{
    size_t j = 0;
    int running = *prior;
    const __m512i zero_i = _mm512_setzero_si512();
    const __m512 veb = _mm512_set1_ps(absErrBound);

    for (; j + 16 <= current_block_size; j += 16) {
        __m512i vmag =
            _mm512_loadu_si512((const __m512i *)(temp_predict_arr + j));
        __m512i vsign =
            _mm512_loadu_si512((const __m512i *)(temp_sign_arr + j));
        __mmask16 neg_mask =
            _mm512_cmp_epi32_mask(vsign, zero_i, _MM_CMPINT_NE);
        __m512i vneg = _mm512_sub_epi32(zero_i, vmag);
        __m512i vdiff = _mm512_mask_blend_epi32(neg_mask, vmag, vneg);
        __m512i vprefix = szp_avx512_prefix_sum_epi32(vdiff);
        __m512 vf;
        __m128i last_lane;

        vprefix = _mm512_add_epi32(vprefix, _mm512_set1_epi32(running));
        vf = _mm512_mul_ps(_mm512_cvtepi32_ps(vprefix), veb);
        _mm512_storeu_ps(newData + j, vf);
        last_lane = _mm512_extracti32x4_epi32(vprefix, 3);
        running = _mm_extract_epi32(last_lane, 3);
    }

    for (; j < current_block_size; j++) {
        int diff = (int)temp_predict_arr[j];
        if (temp_sign_arr[j] != 0U) {
            diff = -diff;
        }
        running += diff;
        newData[j] = (float)running * absErrBound;
    }

    *prior = running;
}
#endif
#endif

void szp_float_decompress_vecBlockaligned_singlepass(float *newData,
                                                     size_t nbEle,
                                                     float absErrBound,
                                                     int blockSize,
                                                     unsigned char *cmpBytes)
{
#ifdef _OPENMP
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp = NULL;
    size_t block_size = (size_t)blockSize;
    unsigned int nbThreads = 0;
    size_t threadblocksize = 0;

#pragma omp parallel
    {
        int tid = 0;
        size_t lo = 0;
        size_t hi = 0;
        float *newData_perthread = NULL;
        int prior = 0;
        unsigned char *block_pointer = NULL;
        unsigned int *temp_sign_arr = NULL;
        unsigned int *temp_predict_arr = NULL;

#pragma omp single
        {
            nbThreads = (unsigned int)omp_get_num_threads();
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            threadblocksize = szp_float_blockaligned_threadblocksize_decode(
                nbEle, block_size, nbThreads);
        }

        tid = omp_get_thread_num();
        lo = (size_t)tid * threadblocksize;
        hi = (size_t)(tid + 1) * threadblocksize;
        if (tid == (int)nbThreads - 1) {
            hi = nbEle;
        }

        newData_perthread = newData + lo;
        block_pointer = rcp + offsets[tid];

        if (lo < hi) {
            float ori_prior = 0.0f;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float));
            newData_perthread += 1;
        }

        temp_sign_arr =
            (unsigned int *)malloc(block_size * sizeof(unsigned int));
        temp_predict_arr =
            (unsigned int *)malloc(block_size * sizeof(unsigned int));

        for (size_t i = lo + 1; i < hi; i += block_size) {
            size_t current_block_size =
                (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) {
                continue;
            }

            unsigned int bit_count = (unsigned int)(*block_pointer++);
            if (bit_count == 0U) {
                float ori_prior = (float)prior * absErrBound;
                szp_float_fill_constant_block(newData_perthread,
                                              current_block_size, ori_prior);
                newData_perthread += current_block_size;
                continue;
            }

            size_t signbytelength = (current_block_size + 7U) >> 3;
#ifdef __AVX2__
            Jiajun_convertByte2UInt_fast_1b_args_avx2(
                current_block_size, block_pointer, signbytelength,
                temp_sign_arr);
#else
            Jiajun_convertByte2UInt_fast_1b_args(
                current_block_size, block_pointer, signbytelength,
                temp_sign_arr);
#endif
            block_pointer += signbytelength;

            size_t savedbitsbytelength = Jiajun_extract_fixed_length_bits(
                block_pointer, current_block_size, temp_predict_arr, bit_count);
            block_pointer += savedbitsbytelength;

#ifdef __AVX512F__
            szp_float_block_vecBlockaligned_singlepass_decode_SIMD512(
                newData_perthread, current_block_size, absErrBound, &prior,
                temp_sign_arr, temp_predict_arr);
#elif defined(__AVX2__)
            szp_float_block_vecBlockaligned_singlepass_decode_SIMD(
                newData_perthread, current_block_size, absErrBound, &prior,
                temp_sign_arr, temp_predict_arr);
#else
            szp_float_block_vecBlockaligned_singlepass_decode_scalar(
                newData_perthread, current_block_size, absErrBound, &prior,
                temp_sign_arr, temp_predict_arr);
#endif
            newData_perthread += current_block_size;
        }

        free(temp_predict_arr);
        free(temp_sign_arr);
    }
#else
    printf("Error! OpenMP not supported!\n");
#endif
}

static inline void szp_float_unpack_sign_bits_1b(size_t count,
                                                 const unsigned char *src,
                                                 unsigned int *dst)
{
#ifdef __AVX2__
    Jiajun_convertByte2UInt_fast_1b_args_avx2(
        count, const_cast<unsigned char *>(src), (count + 7U) >> 3, dst);
#else
    Jiajun_convertByte2UInt_fast_1b_args(
        count, const_cast<unsigned char *>(src), (count + 7U) >> 3, dst);
#endif
}

static inline size_t szp_float_unpack_fixed_bits(const unsigned char *src,
                                                 size_t count,
                                                 unsigned int *dst,
                                                 unsigned int bit_count)
{
    return Jiajun_extract_fixed_length_bits(
        const_cast<unsigned char *>(src), count, dst, bit_count);
}

float *szp_float_decompress_openmp_threadblock_randomaccess(size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP
    float *newData = (float *)malloc(sizeof(float) * nbEle);
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    size_t threadblocksize = 0;
    int block_size = blockSize;

#pragma omp parallel
{
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            threadblocksize = nbEle / nbThreads;
        }
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle;
        }
        float *newData_perthread = newData + lo;
        size_t i = 0;
        size_t j = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *outputBytes_perthread = rcp + offsets[tid]; 
        unsigned char *block_pointer = outputBytes_perthread;

        float ori_prior = 0.0;
        float ori_current = 0.0;

        unsigned char *temp_sign_arr = (unsigned char *)malloc((block_size-1) * sizeof(unsigned char)); // 1 direct value and (block_size - 1) diff. values
        
        unsigned int *temp_predict_arr = (unsigned int *)malloc((block_size-1) * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float)); 
            newData_perthread ++;

            if (current_block_size > 1)
            {
                bit_count = block_pointer[0];
                block_pointer++;

                if (bit_count == 0)
                {
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(current_block_size - 1, block_pointer, (current_block_size - 2) / 8 + 1, temp_sign_arr);
                    block_pointer += ((current_block_size - 2) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size - 1, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        if (temp_sign_arr[j] == 0)
                        {
                            diff = temp_predict_arr[j];
                        }
                        else
                        {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float)current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }
        free(temp_sign_arr);
        free(temp_predict_arr);
    }
    return newData;

#else
    printf("Error! OpenMP not supported!\n");
#endif
}


void szp_float_decompress_openmp_threadblock_randomaccess_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP
    // *newData = (float *)malloc(sizeof(float) * nbEle);
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;
    
    size_t threadblocksize = 0;
    int block_size = blockSize;
#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            rcp = cmpBytes + nbThreads * sizeof(size_t);
            threadblocksize = nbEle / nbThreads;
        }
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle;
        }
        float *newData_perthread = newData + lo;
        size_t i = 0;
        size_t j = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *outputBytes_perthread = rcp + offsets[tid]; 
        unsigned char *block_pointer = outputBytes_perthread;

        float ori_prior = 0.0;
        float ori_current = 0.0;

        
        unsigned char *temp_sign_arr = (unsigned char *)malloc((block_size-1) * sizeof(unsigned char)); // 1 direct value and block_size - 1 diff. values
        
        unsigned int *temp_predict_arr = (unsigned int *)malloc((block_size-1) * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float)); 
            newData_perthread ++;

            if (current_block_size > 1)
            {
                bit_count = block_pointer[0];
                block_pointer++;

                if (bit_count == 0)
                {
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(current_block_size - 1, block_pointer, (current_block_size - 2) / 8 + 1, temp_sign_arr);
                    block_pointer += ((current_block_size - 2) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size - 1, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        if (temp_sign_arr[j] == 0)
                        {
                            diff = temp_predict_arr[j];
                        }
                        else
                        {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float)current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }
        free(temp_sign_arr);
        free(temp_predict_arr);
    }

#else
    printf("Error! OpenMP not supported!\n");
#endif
}

/**
 *  @file szp_float.h
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>, Sheng Di <sdi1@anl.gov>
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include "szp.h"
#include "szp_float.h"
#include <assert.h>
#include <math.h>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
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

int *
szp_float_openmp_direct_predict_quantization(float *oriData, size_t *outSize, float absErrBound,
                                             size_t nbEle, int blockSize)
{
#ifdef _OPENMP

    float *op = oriData;

    size_t i = 0;


    int *quti_arr = (int *)malloc(nbEle * sizeof(int));
    int *diff_arr = (int *)malloc(nbEle * sizeof(int));
    (*outSize) = 0;


    int nbThreads = 1;
    double inver_bound = 1;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            
            inver_bound = 1 / absErrBound;
            
        }

     
#pragma omp for schedule(static)
        for (i = 0; i < nbEle; i++)
        {
            quti_arr[i] = (op[i] + absErrBound) * inver_bound;
        }

#pragma omp single
        {
            diff_arr[0] = quti_arr[0];
        }

#pragma omp for schedule(static)
        for (i = 1; i < nbEle; i++)
        {
            diff_arr[i] = quti_arr[i] - quti_arr[i - 1];
        }
    }

    free(quti_arr);

    return diff_arr;
#else
    return NULL;
#endif
}

int *
szp_float_openmp_threadblock_predict_quantization(float *oriData, size_t *outSize, float absErrBound,
                                                  size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    
    float *op = oriData;

  
    int *diff_arr = (int *)malloc(nbEle * sizeof(int));
    (*outSize) = 0;
    

    int nbThreads = 1;
    double inver_bound = 1;
    int threadblocksize = 1;
    int remainder = 1;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            
            inver_bound = 1 / absErrBound;
            threadblocksize = nbEle / nbThreads;
            remainder = nbEle % nbThreads;
            
        }
        size_t i = 0;
    
        int tid = omp_get_thread_num();
        int lo = tid * threadblocksize;
        int hi = (tid + 1) * threadblocksize;
        int prior = 0;
        int current = 0;
        prior = (op[lo] + absErrBound) * inver_bound;
        diff_arr[lo] = prior;
        for (i = lo + 1; i < hi; i++)
        {
            current = (op[i] + absErrBound) * inver_bound;
            diff_arr[i] = current - prior;
            prior = current;
        }
#pragma omp single
        {
            if (remainder != 0)
            {
                size_t remainder_lo = nbEle - remainder;
                prior = (op[remainder_lo] + absErrBound) * inver_bound;
                diff_arr[remainder_lo] = prior;
                for (i = nbEle - remainder + 1; i < nbEle; i++)
                {
                    current = (op[i] + absErrBound) * inver_bound;
                    diff_arr[i] = current - prior;
                    prior = current;
                }
            }
          
        }
    }

    return diff_arr;
#else
    return NULL;
#endif
}

unsigned char *
szp_float_openmp_threadblock(float *oriData, size_t *outSize, float absErrBound,
                             size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    
    float *op = oriData;

    size_t maxPreservedBufferSize = sizeof(float) * nbEle + sizeof(float);
    size_t maxPreservedBufferSize_perthread = 0;

    unsigned char *real_outputBytes; 
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    unsigned char *output = (unsigned char *)malloc(maxPreservedBufferSize);
    floatToBytes(output, absErrBound); 
    unsigned char *outputBytes = output + sizeof(float); // skip the first buffer for absErrBound
   
    (*outSize) = 0;
  

    unsigned int nbThreads = 0;
    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int block_size = blockSize;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t); 
            outSize_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

            inver_bound = 1 / absErrBound;
            threadblocksize = nbEle / nbThreads;
        }
        size_t i = 0;
        size_t j = 0;
        maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
        unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;
        
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle; // Ensure the last thread processes all remaining elements
        }

        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        
        if (lo < hi) { // Ensure thread has data to process
            prior = (op[lo]) * inver_bound;
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);
        }
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
       
        for (i = lo + 1; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            max = 0;
            for (j = 0; j < current_block_size; j++)
            {
                current = (op[i + j]) * inver_bound;
                diff = current - prior;
                prior = current;
                if (diff == 0)
                {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = 0;
                }
                else
                {
                    if (diff < 0)
                    {
                        temp_sign_arr[j] = 1;
                        temp_predict_arr[j] = -diff;
                    }
                    else
                    {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = diff;
                    }
                    if (max < temp_predict_arr[j])
                        max = temp_predict_arr[j];
                }
            }

            if (max == 0) 
            {
                block_pointer[0] = 0;
                block_pointer++;
                outSize_perthread++;
            }
            else
            {
                bit_count = (int)(log2f(max)) + 1;
                block_pointer[0] = bit_count;
                
                outSize_perthread++;
                block_pointer++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size, block_pointer); 
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                
                savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size, block_pointer, bit_count);
                
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }

        outSize_perthread_arr[tid] = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++)
            {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1];
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
        }
#pragma omp barrier
        memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
        
        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
#pragma omp barrier
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }
    (*outSize) += sizeof(float);
    return output;
#else
    printf("Error! OpenMP not supported!\n");
    return NULL;
#endif
}

/**
 * output: the first 4 bytes are used to store absErrorBound, then followed by compressed data bytes.
 * */
void szp_float_openmp_threadblock_arg(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                      size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    
    float *op = oriData;

    
    size_t maxPreservedBufferSize = sizeof(float) + sizeof(float) * nbEle; 
    size_t maxPreservedBufferSize_perthread = 0;
    
    unsigned char *real_outputBytes; 
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;
    
	unsigned char* outputBytes = output + sizeof(float);
	floatToBytes(output, absErrBound);

    (*outSize) = 0;


    unsigned int nbThreads = 0;
    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int block_size = blockSize;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t); 
            outSize_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

            inver_bound = 1 / absErrBound;
            threadblocksize = nbEle / nbThreads;
        }
        size_t i = 0;
        size_t j = 0;
        maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
        unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;
        
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle; // Ensure the last thread processes all remaining elements
        }

        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        
        if (lo < hi) { // Ensure thread has data to process
            prior = (op[lo]) * inver_bound;
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);
        } // if nbThreads > nbEle, threadblocksize=0, causing no data to be written
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(block_size * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(block_size * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo + 1; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            max = 0;
            for (j = 0; j < current_block_size; j++)
            {
                current = (op[i + j]) * inver_bound;
                diff = current - prior;
                prior = current;
                if (diff == 0)
                {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = 0;
                }
                else
                {
                    if (diff < 0)
                    {
                        temp_sign_arr[j] = 1;
                        temp_predict_arr[j] = -diff;
                    }
                    else
                    {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = diff;
                    }
                    if (max < temp_predict_arr[j])
                        max = temp_predict_arr[j];
                }
            }

            if (max == 0) 
            {
                block_pointer[0] = 0;
                block_pointer++;
                outSize_perthread++;
            }
            else
            {
                bit_count = (int)(log2f(max)) + 1;
                block_pointer[0] = bit_count;
                
                outSize_perthread++;
                block_pointer++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size, block_pointer); 
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                
                savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size, block_pointer, bit_count);
                
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }

        outSize_perthread_arr[tid] = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++)
            {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1];
                
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
            
        }
#pragma omp barrier
        memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
        
        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
#pragma omp barrier
#pragma omp single
        {
            
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }

       
    }
    
    (*outSize) += sizeof(float);

    
#else
    printf("Error! OpenMP not supported!\n");
#endif
}

size_t szp_float_single_thread_arg_buffer(
    unsigned char *__restrict__ output, const float *__restrict__ oriData,
    float absErrBound, size_t nbEle, unsigned char *__restrict__ temp_sign_arr,
    unsigned int *__restrict__ temp_predict_arr,
    int *__restrict__ temp_quant_arr)
{
    floatToBytes(output, absErrBound);
    unsigned char *cmpBytes = output + sizeof(float);
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *block_pointer = cmpBytes + sizeof(size_t);
    double inver_bound = 1.0 / absErrBound;
    size_t outSize = sizeof(float) + sizeof(size_t);

    offsets[0] = 0;
    if (nbEle == 0) {
        return outSize;
    }

    int prior = (int)(oriData[0] * inver_bound);
    memcpy(block_pointer, &prior, sizeof(int));
    block_pointer += sizeof(unsigned int);
    outSize += sizeof(unsigned int);

    if (nbEle > 1) {
        size_t current_block_size = nbEle - 1;
        unsigned int max = 0U;
        int *temp_int_predict_arr = reinterpret_cast<int *>(temp_predict_arr);

        for (size_t j = 0; j < current_block_size; j++) {
            temp_quant_arr[j] = (int)(oriData[j + 1] * inver_bound);
        }
        temp_int_predict_arr[0] = temp_quant_arr[0] - prior;
        for (size_t j = 1; j < current_block_size; j++) {
            temp_int_predict_arr[j] = temp_quant_arr[j] - temp_quant_arr[j - 1];
        }
        for (size_t j = 0; j < current_block_size; j++) {
            int diff = temp_int_predict_arr[j];
            if (diff < 0) {
                temp_sign_arr[j] = 1;
                diff = -diff;
            } else {
                temp_sign_arr[j] = 0;
            }
            temp_int_predict_arr[j] = diff;
            if ((unsigned int)diff > max) {
                max = (unsigned int)diff;
            }
        }

        if (max == 0U) {
            *block_pointer++ = 0U;
            outSize += 1;
        } else {
            unsigned int bit_count = (unsigned int)(log2f((float)max)) + 1U;
            *block_pointer++ = (unsigned char)bit_count;
            outSize += 1;

            unsigned int signbytelength = (unsigned int)
                convertIntArray2ByteArray_fast_1b_args(
                    temp_sign_arr, current_block_size, block_pointer);
            block_pointer += signbytelength;
            outSize += signbytelength;

            unsigned int savedbitsbytelength = (unsigned int)
                Jiajun_save_fixed_length_bits(
                    temp_predict_arr, current_block_size, block_pointer,
                    bit_count);
            block_pointer += savedbitsbytelength;
            outSize += savedbitsbytelength;
        }
    }

    return outSize;
}

void szp_float_single_thread_arg(unsigned char *output, float *oriData,
                                 size_t *outSize, float absErrBound,
                                 size_t nbEle, int blockSize)
{
    float *op = oriData;
    unsigned char *outputBytes = output + sizeof(float);
    unsigned char *real_outputBytes = NULL;
    size_t *outSize_perthread_arr = NULL;
    size_t *offsets_perthread_arr = NULL;
    double inver_bound = 1.0 / absErrBound;
    unsigned int block_size = (unsigned int)blockSize;
    int nbThreads = 1;
    size_t maxPreservedBufferSize_perthread = sizeof(float) * nbEle;
    unsigned char *outputBytes_perthread =
        (unsigned char *)malloc(maxPreservedBufferSize_perthread == 0 ? 1
                                                                      : maxPreservedBufferSize_perthread);
    size_t outSize_perthread = 0;
    int prior = 0;
    int current = 0;
    int diff = 0;
    unsigned int max = 0;
    unsigned int bit_count = 0;
    unsigned char *block_pointer = outputBytes_perthread;

    floatToBytes(output, absErrBound);
    (*outSize) = 0;

    real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
    (*outSize) += nbThreads * sizeof(size_t);
    outSize_perthread_arr = (size_t *)malloc(sizeof(size_t));
    offsets_perthread_arr = (size_t *)malloc(sizeof(size_t));

    if (nbEle > 0) {
        prior = (int)(op[0] * inver_bound);
        memcpy(block_pointer, &prior, sizeof(int));
        block_pointer += sizeof(unsigned int);
        outSize_perthread += sizeof(unsigned int);
    }

    unsigned char *temp_sign_arr =
        (unsigned char *)malloc(block_size * sizeof(unsigned char));
    unsigned int *temp_predict_arr =
        (unsigned int *)malloc(block_size * sizeof(unsigned int));
    unsigned int signbytelength = 0;
    unsigned int savedbitsbytelength = 0;

    for (size_t i = 1; i < nbEle; i += block_size) {
        size_t current_block_size =
            (i + block_size > nbEle) ? (nbEle - i) : block_size;
        if (current_block_size == 0) {
            continue;
        }

        max = 0;
        for (size_t j = 0; j < current_block_size; j++) {
            current = (int)(op[i + j] * inver_bound);
            diff = current - prior;
            prior = current;
            if (diff == 0) {
                temp_sign_arr[j] = 0;
                temp_predict_arr[j] = 0;
            } else {
                if (diff < 0) {
                    temp_sign_arr[j] = 1;
                    temp_predict_arr[j] = (unsigned int)(-diff);
                } else {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = (unsigned int)diff;
                }
                if (max < temp_predict_arr[j]) {
                    max = temp_predict_arr[j];
                }
            }
        }

        if (max == 0U) {
            *block_pointer++ = 0U;
            outSize_perthread++;
        } else {
            bit_count = (unsigned int)(log2f((float)max)) + 1U;
            *block_pointer++ = (unsigned char)bit_count;
            outSize_perthread++;
            signbytelength = (unsigned int)convertIntArray2ByteArray_fast_1b_args(
                temp_sign_arr, current_block_size, block_pointer);
            block_pointer += signbytelength;
            outSize_perthread += signbytelength;
            savedbitsbytelength = (unsigned int)Jiajun_save_fixed_length_bits(
                temp_predict_arr, current_block_size, block_pointer, bit_count);
            block_pointer += savedbitsbytelength;
            outSize_perthread += savedbitsbytelength;
        }
    }

    outSize_perthread_arr[0] = outSize_perthread;
    offsets_perthread_arr[0] = 0;
    (*outSize) += outSize_perthread_arr[0];
    memcpy(outputBytes, offsets_perthread_arr, sizeof(size_t));
    memcpy(real_outputBytes, outputBytes_perthread, outSize_perthread);

    free(outputBytes_perthread);
    free(temp_sign_arr);
    free(temp_predict_arr);
    free(outSize_perthread_arr);
    free(offsets_perthread_arr);

    (*outSize) += sizeof(float);
}

size_t szp_float_single_thread_arg_record(unsigned char *output, float *oriData,
                                          size_t *outSize, float absErrBound,
                                          size_t nbEle, int blockSize)
{
    size_t total_memaccess = 0;
    float *op = oriData;
    unsigned char *outputBytes = output + sizeof(float);
    unsigned char *real_outputBytes = NULL;
    size_t *outSize_perthread_arr = NULL;
    size_t *offsets_perthread_arr = NULL;
    double inver_bound = 1.0 / absErrBound;
    unsigned int block_size = (unsigned int)blockSize;
    int nbThreads = 1;
    size_t maxPreservedBufferSize_perthread = sizeof(float) * nbEle;
    unsigned char *outputBytes_perthread =
        (unsigned char *)malloc(maxPreservedBufferSize_perthread == 0 ? 1
                                                                      : maxPreservedBufferSize_perthread);
    size_t outSize_perthread = 0;
    int prior = 0;
    int current = 0;
    int diff = 0;
    unsigned int max = 0;
    unsigned int bit_count = 0;
    unsigned char *block_pointer = outputBytes_perthread;

    floatToBytes(output, absErrBound);
    (*outSize) = 0;
    total_memaccess += sizeof(float);

    real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
    (*outSize) += nbThreads * sizeof(size_t);
    outSize_perthread_arr = (size_t *)malloc(sizeof(size_t));
    offsets_perthread_arr = (size_t *)malloc(sizeof(size_t));

    if (nbEle > 0) {
        prior = (int)(op[0] * inver_bound);
        total_memaccess += sizeof(float);
        memcpy(block_pointer, &prior, sizeof(int));
        total_memaccess += sizeof(int) * 2;
        block_pointer += sizeof(unsigned int);
        outSize_perthread += sizeof(unsigned int);
    }

    unsigned char *temp_sign_arr =
        (unsigned char *)malloc(block_size * sizeof(unsigned char));
    unsigned int *temp_predict_arr =
        (unsigned int *)malloc(block_size * sizeof(unsigned int));
    unsigned int signbytelength = 0;
    unsigned int savedbitsbytelength = 0;

    for (size_t i = 1; i < nbEle; i += block_size) {
        size_t current_block_size =
            (i + block_size > nbEle) ? (nbEle - i) : block_size;
        if (current_block_size == 0) {
            continue;
        }

        max = 0U;
        for (size_t j = 0; j < current_block_size; j++) {
            current = (int)(op[i + j] * inver_bound);
            total_memaccess += sizeof(float);
            diff = current - prior;
            prior = current;
            if (diff == 0) {
                temp_sign_arr[j] = 0;
                temp_predict_arr[j] = 0;
            } else {
                if (diff < 0) {
                    temp_sign_arr[j] = 1;
                    temp_predict_arr[j] = (unsigned int)(-diff);
                } else {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = (unsigned int)diff;
                }
                if (max < temp_predict_arr[j]) {
                    max = temp_predict_arr[j];
                }
            }
            total_memaccess += sizeof(unsigned char) + sizeof(unsigned int);
        }

        if (max == 0U) {
            *block_pointer++ = 0U;
            total_memaccess += sizeof(unsigned char);
            outSize_perthread++;
        } else {
            bit_count = (unsigned int)(log2f((float)max)) + 1U;
            *block_pointer++ = (unsigned char)bit_count;
            total_memaccess += sizeof(unsigned char);
            outSize_perthread++;
            signbytelength = (unsigned int)convertIntArray2ByteArray_fast_1b_args(
                temp_sign_arr, current_block_size, block_pointer);
            total_memaccess += sizeof(unsigned char) * current_block_size;
            block_pointer += signbytelength;
            total_memaccess += sizeof(unsigned char) * signbytelength;
            outSize_perthread += signbytelength;
            savedbitsbytelength = (unsigned int)Jiajun_save_fixed_length_bits(
                temp_predict_arr, current_block_size, block_pointer, bit_count);
            total_memaccess += sizeof(unsigned int) * current_block_size;
            block_pointer += savedbitsbytelength;
            total_memaccess += sizeof(unsigned char) * savedbitsbytelength;
            outSize_perthread += savedbitsbytelength;
        }
    }

    outSize_perthread_arr[0] = outSize_perthread;
    total_memaccess += sizeof(size_t);
    offsets_perthread_arr[0] = 0;
    total_memaccess += sizeof(size_t);
    (*outSize) += outSize_perthread;
    total_memaccess += sizeof(size_t) * 3;
    memcpy(outputBytes, offsets_perthread_arr, sizeof(size_t));
    total_memaccess += sizeof(size_t) * 2;
    memcpy(real_outputBytes, outputBytes_perthread, outSize_perthread);
    total_memaccess += sizeof(unsigned char) * outSize_perthread * 2;

    free(outputBytes_perthread);
    free(temp_sign_arr);
    free(temp_predict_arr);
    free(outSize_perthread_arr);
    free(offsets_perthread_arr);

    (*outSize) += sizeof(float);
    return total_memaccess;
}

static size_t szp_float_block_compiler_buffer(
    unsigned char *__restrict__ block_pointer, const float *__restrict__ op,
    double inver_bound, size_t current_block_size, int *__restrict__ prior,
    unsigned char *__restrict__ temp_sign_arr,
    unsigned int *__restrict__ temp_predict_arr,
    int *__restrict__ temp_quant_arr)
{
    unsigned char *start_block_pointer = block_pointer;
    unsigned int bit_count = 0;

    for (size_t j = 0; j < current_block_size; j++) {
        temp_quant_arr[j] = (int)(op[j] * inver_bound);
    }

    int *temp_int_predict_arr = reinterpret_cast<int *>(temp_predict_arr);
    temp_int_predict_arr[0] = temp_quant_arr[0] - *prior;
    for (size_t j = 1; j < current_block_size; j++) {
        temp_int_predict_arr[j] = temp_quant_arr[j] - temp_quant_arr[j - 1];
    }

    int max = 0;
    for (size_t j = 0; j < current_block_size; j++) {
        int diff = temp_int_predict_arr[j];
        temp_sign_arr[j] = diff < 0 ? 1 : 0;
        diff = abs(diff);
        temp_int_predict_arr[j] = diff;
        max = (diff > max) ? diff : max;
    }

    *prior = temp_quant_arr[current_block_size - 1];

    if (max == 0) {
        block_pointer[0] = 0;
        block_pointer++;
    } else {
#if defined(__GNUC__) || defined(__clang__)
        bit_count = (unsigned int)(sizeof(unsigned int) * 8 - __builtin_clz(max));
#else
        bit_count = (unsigned int)(log2f((float)max)) + 1;
#endif
        block_pointer[0] = (unsigned char)bit_count;
        block_pointer++;

        unsigned int signbytelength = convertIntArray2ByteArray_fast_1b_args(
            temp_sign_arr, current_block_size, block_pointer);
        block_pointer += signbytelength;

        unsigned int savedbitsbytelength = Jiajun_save_fixed_length_bits(
            temp_predict_arr, current_block_size, block_pointer, bit_count);
        block_pointer += savedbitsbytelength;
    }

    return (size_t)(block_pointer - start_block_pointer);
}

void szp_float_openmp_threadblock_arg_buffer(unsigned char *output, float *oriData,
                                             size_t *outSize, float absErrBound,
                                             size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    const float *op = oriData;
    unsigned char *real_outputBytes = NULL;
    size_t *outSize_perthread_arr = NULL;
    size_t *offsets_perthread_arr = NULL;
    unsigned char *outputBytes = output + sizeof(float);
    unsigned int nbThreads = 0;
    double inver_bound = 0;
    size_t threadblocksize = 0;
    unsigned int block_size = (unsigned int)blockSize;

    floatToBytes(output, absErrBound);
    (*outSize) = 0;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t);
            outSize_perthread_arr =
                (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr =
                (size_t *)malloc(nbThreads * sizeof(size_t));
            inver_bound = 1 / absErrBound;
            threadblocksize = (nbEle + nbThreads - 1) / nbThreads;
        }

        int tid = omp_get_thread_num();
        size_t lo = (size_t)tid * threadblocksize;
        size_t hi = (size_t)(tid + 1) * threadblocksize;
        if (hi > nbEle) {
            hi = nbEle;
        }

        size_t local_count = (hi > lo) ? (hi - lo) : 0U;
        size_t max_buffer_size = sizeof(unsigned int) + local_count * (sizeof(float) + 1U);
        unsigned char *outputBytes_perthread =
            (unsigned char *)malloc(max_buffer_size == 0 ? 1 : max_buffer_size);
        size_t outSize_perthread = 0;
        unsigned char *block_pointer = outputBytes_perthread;

        if (lo < hi) {
            unsigned char *temp_sign_arr =
                (unsigned char *)malloc(block_size * sizeof(unsigned char));
            unsigned int *temp_predict_arr =
                (unsigned int *)malloc(block_size * sizeof(unsigned int));
            int *temp_quant_arr = (int *)malloc(block_size * sizeof(int));

            int prior = (int)(op[lo] * inver_bound);
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);

            for (size_t i = lo + 1; i < hi; i += block_size) {
                size_t current_block_size =
                    (i + block_size > hi) ? (hi - i) : block_size;
                if (current_block_size == 0) {
                    continue;
                }

                size_t compressed_block_size = szp_float_block_compiler_buffer(
                    block_pointer, op + i, inver_bound, current_block_size,
                    &prior, temp_sign_arr, temp_predict_arr, temp_quant_arr);
                block_pointer += compressed_block_size;
                outSize_perthread += compressed_block_size;
            }

            free(temp_sign_arr);
            free(temp_predict_arr);
            free(temp_quant_arr);
        }

        outSize_perthread_arr[tid] = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (size_t i = 1; i < nbThreads; i++) {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] +
                                           outSize_perthread_arr[i - 1];
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] +
                          outSize_perthread_arr[nbThreads - 1];
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
        }
#pragma omp barrier
        if (lo < hi) {
            memcpy(real_outputBytes + offsets_perthread_arr[tid],
                   outputBytes_perthread, outSize_perthread);
        }

        free(outputBytes_perthread);
#pragma omp barrier
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }

    (*outSize) += sizeof(float);
#else
    printf("Error! OpenMP not supported!\n");
#endif
}

static inline size_t szp_float_blockaligned_threadblocksize(size_t nbEle,
                                                            size_t block_size,
                                                            unsigned int nbThreads)
{
    if (block_size == 0 || nbThreads == 0) {
        return 0;
    }
    return (nbEle / block_size / nbThreads) * block_size;
}

static inline size_t szp_float_blockaligned_buffer_capacity(size_t element_count)
{
    return sizeof(unsigned int) + element_count * (sizeof(float) + 1U);
}

static inline unsigned int szp_float_abs_i32_to_u32(int value)
{
    int mask = value >> 31;
    return (unsigned int)((value ^ mask) - mask);
}

static inline size_t szp_float_emit_blockaligned_block(
    unsigned char *block_pointer, size_t current_block_size,
    const unsigned int *temp_sign_arr, const unsigned int *temp_predict_arr,
    unsigned int max_value)
{
    unsigned char *start_block_pointer = block_pointer;

    if (max_value == 0U) {
        *block_pointer++ = 0;
        return (size_t)(block_pointer - start_block_pointer);
    }

    unsigned int bit_count = 0U;
#if defined(__GNUC__) || defined(__clang__)
    bit_count =
        (unsigned int)(sizeof(unsigned int) * 8U - __builtin_clz(max_value));
#else
    bit_count = (unsigned int)(log2f((float)max_value)) + 1U;
#endif

    *block_pointer++ = (unsigned char)bit_count;
    block_pointer += Jiajun_convertUInt2Byte_fast_1b_args_MSB_SIMD(
        (unsigned int *)temp_sign_arr, current_block_size, block_pointer);
    block_pointer += Jiajun_save_fixed_length_bits_MSB(
        (unsigned int *)temp_predict_arr, current_block_size, block_pointer,
        bit_count);
    return (size_t)(block_pointer - start_block_pointer);
}

static size_t szp_float_blockaligned_scalar_worker(
    unsigned char *block_pointer, const float *op, float inver_bound,
    size_t current_block_size, int *prior, unsigned int *temp_sign_arr,
    unsigned int *temp_predict_arr, int *temp_quant_arr)
{
    int *temp_int_predict_arr = reinterpret_cast<int *>(temp_predict_arr);
    unsigned int max_value = 0U;

    for (size_t j = 0; j < current_block_size; j++) {
        temp_quant_arr[j] = (int)(op[j] * inver_bound);
    }

    temp_int_predict_arr[0] = temp_quant_arr[0] - *prior;
    for (size_t j = 1; j < current_block_size; j++) {
        temp_int_predict_arr[j] = temp_quant_arr[j] - temp_quant_arr[j - 1];
    }

    for (size_t j = 0; j < current_block_size; j++) {
        int diff = temp_int_predict_arr[j];
        int mask = diff >> 31;
        unsigned int abs_diff = (unsigned int)((diff ^ mask) - mask);

        temp_sign_arr[j] = (unsigned int)(mask & 1);
        temp_int_predict_arr[j] = (int)abs_diff;
        if (abs_diff > max_value) {
            max_value = abs_diff;
        }
    }

    *prior = temp_quant_arr[current_block_size - 1];
    return szp_float_emit_blockaligned_block(
        block_pointer, current_block_size, temp_sign_arr, temp_predict_arr,
        max_value);
}

#if defined(__AVX2__) || defined(__AVX512F__)
static inline int szp_float_avx2_horizontal_max_epi32(__m256i v)
{
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i max128 = _mm_max_epi32(hi, lo);
    __m128i max2 = _mm_shuffle_epi32(max128, _MM_SHUFFLE(1, 0, 3, 2));

    max128 = _mm_max_epi32(max128, max2);
    max2 = _mm_shuffle_epi32(max128, _MM_SHUFFLE(2, 3, 0, 1));
    max128 = _mm_max_epi32(max128, max2);
    return _mm_extract_epi32(max128, 0);
}

static inline unsigned int szp_float_avx2_horizontal_max_epu32(__m256i v)
{
    unsigned int vals[8];

    _mm256_storeu_si256((__m256i *)vals, v);
    unsigned int max_value = vals[0];
    for (int i = 1; i < 8; i++) {
        if (vals[i] > max_value) {
            max_value = vals[i];
        }
    }
    return max_value;
}

#ifdef __AVX512F__
static inline int szp_float_avx512_horizontal_max_epi32(__m512i v)
{
    int vals[16];

    _mm512_storeu_si512((__m512i *)vals, v);
    int max_value = vals[0];
    for (int i = 1; i < 16; i++) {
        if (vals[i] > max_value) {
            max_value = vals[i];
        }
    }
    return max_value;
}

static inline unsigned int szp_float_avx512_horizontal_max_epu32(__m512i v)
{
    unsigned int vals[16];

    _mm512_storeu_si512((__m512i *)vals, v);
    unsigned int max_value = vals[0];
    for (int i = 1; i < 16; i++) {
        if (vals[i] > max_value) {
            max_value = vals[i];
        }
    }
    return max_value;
}

static size_t szp_float_vec_blockaligned_simd512_worker(
    unsigned char *block_pointer, const float *op, float inver_bound,
    size_t current_block_size, int *prior, unsigned int *temp_sign_arr,
    unsigned int *temp_predict_arr, int *temp_quant_arr)
{
    size_t j = 0;
    const int prev = *prior;
    size_t simd_count = current_block_size & ~15UL;
    __m512 v_inver_bound = _mm512_set1_ps(inver_bound);
    __m512i v_max = _mm512_setzero_si512();
    __m512i v_zero = _mm512_setzero_si512();
    int *temp_int_predict_arr = reinterpret_cast<int *>(temp_predict_arr);

    for (j = 0; j < simd_count; j += 16) {
        __m512 v_data = _mm512_loadu_ps(op + j);
        __m512i v_quant =
            _mm512_cvttps_epi32(_mm512_mul_ps(v_data, v_inver_bound));
        _mm512_storeu_si512((__m512i *)(temp_quant_arr + j), v_quant);
    }
    for (; j < current_block_size; j++) {
        temp_quant_arr[j] = (int)(op[j] * inver_bound);
    }

    temp_int_predict_arr[0] = temp_quant_arr[0] - prev;
    for (j = 1; j < simd_count; j += 16) {
        if (j + 15 >= current_block_size) {
            break;
        }
        __m512i v_curr =
            _mm512_loadu_si512((const __m512i *)(temp_quant_arr + j));
        __m512i v_prev =
            _mm512_loadu_si512((const __m512i *)(temp_quant_arr + j - 1));
        __m512i v_diff = _mm512_sub_epi32(v_curr, v_prev);
        _mm512_storeu_si512((__m512i *)(temp_int_predict_arr + j), v_diff);
    }
    for (; j < current_block_size; j++) {
        temp_int_predict_arr[j] = temp_quant_arr[j] - temp_quant_arr[j - 1];
    }

    for (j = 0; j < simd_count; j += 16) {
        __m512i v_diff =
            _mm512_loadu_si512((const __m512i *)(temp_int_predict_arr + j));
        __mmask16 neg_mask =
            _mm512_cmp_epi32_mask(v_diff, v_zero, _MM_CMPINT_LT);
        __m512i v_sign = _mm512_maskz_set1_epi32(neg_mask, 1);
        __m512i v_neg = _mm512_sub_epi32(v_zero, v_diff);
        __m512i v_abs = _mm512_mask_blend_epi32(neg_mask, v_diff, v_neg);

        v_max = _mm512_max_epi32(v_max, v_abs);
        _mm512_storeu_si512((__m512i *)(temp_int_predict_arr + j), v_abs);
        _mm512_storeu_si512((__m512i *)(temp_sign_arr + j), v_sign);
    }

    unsigned int max_value =
        (unsigned int)szp_float_avx512_horizontal_max_epi32(v_max);
    for (; j < current_block_size; j++) {
        int diff = temp_int_predict_arr[j];
        unsigned int mag = szp_float_abs_i32_to_u32(diff);

        temp_sign_arr[j] = (unsigned int)((unsigned int)diff >> 31);
        temp_predict_arr[j] = mag;
        if (mag > max_value) {
            max_value = mag;
        }
    }

    *prior = temp_quant_arr[current_block_size - 1];
    return szp_float_emit_blockaligned_block(
        block_pointer, current_block_size, temp_sign_arr, temp_predict_arr,
        max_value);
}
#endif

static size_t szp_float_vec_blockaligned_simd_worker(
    unsigned char *block_pointer, const float *op, float inver_bound,
    size_t current_block_size, int *prior, unsigned int *temp_sign_arr,
    unsigned int *temp_predict_arr, int *temp_quant_arr)
{
    size_t j = 0;
    const int prev = *prior;
    size_t simd_count = current_block_size & ~7UL;
    __m256 v_inver_bound = _mm256_set1_ps(inver_bound);
    __m256i v_max = _mm256_setzero_si256();
    __m256i v_zero = _mm256_setzero_si256();
    __m256i v_one = _mm256_set1_epi32(1);
    int *temp_int_predict_arr = reinterpret_cast<int *>(temp_predict_arr);

    for (j = 0; j < simd_count; j += 8) {
        __m256 v_data = _mm256_loadu_ps(op + j);
        __m256i v_quant =
            _mm256_cvttps_epi32(_mm256_mul_ps(v_data, v_inver_bound));
        _mm256_storeu_si256((__m256i *)(temp_quant_arr + j), v_quant);
    }
    for (; j < current_block_size; j++) {
        temp_quant_arr[j] = (int)(op[j] * inver_bound);
    }

    temp_int_predict_arr[0] = temp_quant_arr[0] - prev;
    for (j = 1; j < simd_count; j += 8) {
        if (j + 7 >= current_block_size) {
            break;
        }
        __m256i v_curr =
            _mm256_loadu_si256((const __m256i *)(temp_quant_arr + j));
        __m256i v_prev =
            _mm256_loadu_si256((const __m256i *)(temp_quant_arr + j - 1));
        __m256i v_diff = _mm256_sub_epi32(v_curr, v_prev);
        _mm256_storeu_si256((__m256i *)(temp_int_predict_arr + j), v_diff);
    }
    for (; j < current_block_size; j++) {
        temp_int_predict_arr[j] = temp_quant_arr[j] - temp_quant_arr[j - 1];
    }

    for (j = 0; j < simd_count; j += 8) {
        __m256i v_diff =
            _mm256_loadu_si256((const __m256i *)(temp_int_predict_arr + j));
        __m256i v_sign_mask = _mm256_cmpgt_epi32(v_zero, v_diff);
        __m256i v_sign = _mm256_and_si256(v_sign_mask, v_one);
        __m256i v_abs = _mm256_abs_epi32(v_diff);

        v_max = _mm256_max_epi32(v_max, v_abs);
        _mm256_storeu_si256((__m256i *)(temp_int_predict_arr + j), v_abs);
        _mm256_storeu_si256((__m256i *)(temp_sign_arr + j), v_sign);
    }

    unsigned int max_value =
        (unsigned int)szp_float_avx2_horizontal_max_epi32(v_max);
    for (; j < current_block_size; j++) {
        int diff = temp_int_predict_arr[j];
        unsigned int mag = szp_float_abs_i32_to_u32(diff);

        temp_sign_arr[j] = (unsigned int)((unsigned int)diff >> 31);
        temp_predict_arr[j] = mag;
        if (mag > max_value) {
            max_value = mag;
        }
    }

    *prior = temp_quant_arr[current_block_size - 1];
    return szp_float_emit_blockaligned_block(
        block_pointer, current_block_size, temp_sign_arr, temp_predict_arr,
        max_value);
}

static size_t szp_float_vec_blockaligned_singlepass_simd_worker(
    unsigned char *block_pointer, const float *op, float inver_bound,
    size_t current_block_size, int *prior, unsigned int *temp_sign_arr,
    unsigned int *temp_predict_arr, int *temp_quant_arr)
{
    (void)temp_quant_arr;
    size_t j = 0;
    int prior_quant = *prior;
    unsigned int max_value = 0U;
    const __m256 v_inver_bound = _mm256_set1_ps(inver_bound);
    const __m256i v_one = _mm256_set1_epi32(1);
    const __m256i shift_idx = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
    __m256i v_max = _mm256_setzero_si256();

    for (; j + 8 <= current_block_size; j += 8) {
        __m256 v_data = _mm256_loadu_ps(op + j);
        __m256i v_quant =
            _mm256_cvttps_epi32(_mm256_mul_ps(v_data, v_inver_bound));
        __m256i v_prev =
            _mm256_permutevar8x32_epi32(v_quant, shift_idx);
        __m256i v_diff;
        __m256i v_mask;
        __m256i v_abs;

        v_prev = _mm256_blend_epi32(v_prev, _mm256_set1_epi32(prior_quant),
                                    0x01);
        v_diff = _mm256_sub_epi32(v_quant, v_prev);
        v_mask = _mm256_srai_epi32(v_diff, 31);
        v_abs = _mm256_sub_epi32(_mm256_xor_si256(v_diff, v_mask), v_mask);
        v_max = _mm256_max_epu32(v_max, v_abs);
        _mm256_storeu_si256((__m256i *)(temp_sign_arr + j),
                            _mm256_and_si256(v_mask, v_one));
        _mm256_storeu_si256((__m256i *)(temp_predict_arr + j), v_abs);
        prior_quant = _mm256_extract_epi32(v_quant, 7);
    }

    if (j != 0) {
        max_value = szp_float_avx2_horizontal_max_epu32(v_max);
    }

    for (; j < current_block_size; j++) {
        int quant = (int)(op[j] * inver_bound);
        int diff = quant - prior_quant;
        unsigned int mag = szp_float_abs_i32_to_u32(diff);

        temp_sign_arr[j] = (unsigned int)((unsigned int)diff >> 31);
        temp_predict_arr[j] = mag;
        if (mag > max_value) {
            max_value = mag;
        }
        prior_quant = quant;
    }

    *prior = prior_quant;
    return szp_float_emit_blockaligned_block(
        block_pointer, current_block_size, temp_sign_arr, temp_predict_arr,
        max_value);
}

#ifdef __AVX512F__
static size_t szp_float_vec_blockaligned_singlepass_simd512_worker(
    unsigned char *block_pointer, const float *op, float inver_bound,
    size_t current_block_size, int *prior, unsigned int *temp_sign_arr,
    unsigned int *temp_predict_arr, int *temp_quant_arr)
{
    (void)temp_quant_arr;
    size_t j = 0;
    int prior_quant = *prior;
    unsigned int max_value = 0U;
    const __m512 v_inver_bound = _mm512_set1_ps(inver_bound);
    const __m512i v_one = _mm512_set1_epi32(1);
    __m512i v_max = _mm512_setzero_si512();
    const __m512i shift_idx = _mm512_setr_epi32(
        15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);

    for (; j + 16 <= current_block_size; j += 16) {
        __m512 v_data = _mm512_loadu_ps(op + j);
        __m512i v_quant =
            _mm512_cvttps_epi32(_mm512_mul_ps(v_data, v_inver_bound));
        __m512i v_prev = _mm512_permutexvar_epi32(shift_idx, v_quant);
        __m512i v_diff;
        __m512i v_mask;
        __m512i v_abs;
        __m128i last_lane;

        v_prev = _mm512_mask_blend_epi32(
            0x0001, v_prev, _mm512_set1_epi32(prior_quant));
        v_diff = _mm512_sub_epi32(v_quant, v_prev);
        v_mask = _mm512_srai_epi32(v_diff, 31);
        v_abs = _mm512_sub_epi32(_mm512_xor_si512(v_diff, v_mask), v_mask);
        v_max = _mm512_max_epu32(v_max, v_abs);
        _mm512_storeu_si512((__m512i *)(temp_sign_arr + j),
                            _mm512_and_si512(v_mask, v_one));
        _mm512_storeu_si512((__m512i *)(temp_predict_arr + j), v_abs);

        last_lane = _mm512_extracti32x4_epi32(v_quant, 3);
        prior_quant = _mm_extract_epi32(last_lane, 3);
    }

    if (j != 0) {
        max_value = szp_float_avx512_horizontal_max_epu32(v_max);
    }

    for (; j < current_block_size; j++) {
        int quant = (int)(op[j] * inver_bound);
        int diff = quant - prior_quant;
        unsigned int mag = szp_float_abs_i32_to_u32(diff);

        temp_sign_arr[j] = (unsigned int)((unsigned int)diff >> 31);
        temp_predict_arr[j] = mag;
        if (mag > max_value) {
            max_value = mag;
        }
        prior_quant = quant;
    }

    *prior = prior_quant;
    return szp_float_emit_blockaligned_block(
        block_pointer, current_block_size, temp_sign_arr, temp_predict_arr,
        max_value);
}
#endif
#endif

static size_t szp_float_vec_blockaligned_singlepass_scalar_worker(
    unsigned char *block_pointer, const float *op, float inver_bound,
    size_t current_block_size, int *prior, unsigned int *temp_sign_arr,
    unsigned int *temp_predict_arr, int *temp_quant_arr)
{
    (void)temp_quant_arr;
    unsigned int max_value = 0U;
    int prior_quant = *prior;

    for (size_t j = 0; j < current_block_size; j++) {
        int quant = (int)(op[j] * inver_bound);
        int diff = quant - prior_quant;
        unsigned int mag = szp_float_abs_i32_to_u32(diff);

        temp_sign_arr[j] = (unsigned int)((unsigned int)diff >> 31);
        temp_predict_arr[j] = mag;
        if (mag > max_value) {
            max_value = mag;
        }
        prior_quant = quant;
    }

    *prior = prior_quant;
    return szp_float_emit_blockaligned_block(
        block_pointer, current_block_size, temp_sign_arr, temp_predict_arr,
        max_value);
}

static size_t szp_float_vec_blockaligned_worker(
    unsigned char *block_pointer, const float *op, float inver_bound,
    size_t current_block_size, int *prior, unsigned int *temp_sign_arr,
    unsigned int *temp_predict_arr, int *temp_quant_arr)
{
#ifdef __AVX512F__
    return szp_float_vec_blockaligned_simd512_worker(
        block_pointer, op, inver_bound, current_block_size, prior,
        temp_sign_arr, temp_predict_arr, temp_quant_arr);
#elif defined(__AVX2__)
    return szp_float_vec_blockaligned_simd_worker(
        block_pointer, op, inver_bound, current_block_size, prior,
        temp_sign_arr, temp_predict_arr, temp_quant_arr);
#else
    return szp_float_blockaligned_scalar_worker(
        block_pointer, op, inver_bound, current_block_size, prior,
        temp_sign_arr, temp_predict_arr, temp_quant_arr);
#endif
}

static size_t szp_float_vec_blockaligned_singlepass_worker(
    unsigned char *block_pointer, const float *op, float inver_bound,
    size_t current_block_size, int *prior, unsigned int *temp_sign_arr,
    unsigned int *temp_predict_arr, int *temp_quant_arr)
{
#ifdef __AVX512F__
    return szp_float_vec_blockaligned_singlepass_simd512_worker(
        block_pointer, op, inver_bound, current_block_size, prior,
        temp_sign_arr, temp_predict_arr, temp_quant_arr);
#elif defined(__AVX2__)
    return szp_float_vec_blockaligned_singlepass_simd_worker(
        block_pointer, op, inver_bound, current_block_size, prior,
        temp_sign_arr, temp_predict_arr, temp_quant_arr);
#else
    return szp_float_vec_blockaligned_singlepass_scalar_worker(
        block_pointer, op, inver_bound, current_block_size, prior,
        temp_sign_arr, temp_predict_arr, temp_quant_arr);
#endif
}

typedef size_t (*szp_float_blockaligned_worker_fn)(
    unsigned char *, const float *, float, size_t, int *, unsigned int *,
    unsigned int *, int *);

static void szp_float_openmp_blockaligned_impl(
    unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize, szp_float_blockaligned_worker_fn worker)
{
#ifdef _OPENMP
    float *op = oriData;
    unsigned char *outputBytes = output + sizeof(float);
    unsigned char *real_outputBytes = NULL;
    size_t *outSize_perthread_arr = NULL;
    size_t *offsets_perthread_arr = NULL;
    unsigned int nbThreads = 0;
    size_t threadblocksize = 0;
    unsigned int block_size = (unsigned int)blockSize;
    float inver_bound = 1.0f / absErrBound;

    floatToBytes(output, absErrBound);
    *outSize = 0;
    if (block_size == 0U) {
        *outSize = sizeof(float);
        return;
    }

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            *outSize += nbThreads * sizeof(size_t);
            outSize_perthread_arr =
                (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr =
                (size_t *)malloc(nbThreads * sizeof(size_t));
            threadblocksize = szp_float_blockaligned_threadblocksize(
                nbEle, block_size, nbThreads);
        }

        int tid = omp_get_thread_num();
        size_t lo = (size_t)tid * threadblocksize;
        size_t hi = (size_t)(tid + 1) * threadblocksize;
        if (tid == (int)nbThreads - 1) {
            hi = nbEle;
        }

        size_t local_count = (hi > lo) ? (hi - lo) : 0U;
        size_t max_buffer_size =
            szp_float_blockaligned_buffer_capacity(local_count);
        unsigned char *outputBytes_perthread =
            (unsigned char *)malloc(max_buffer_size == 0 ? 1 : max_buffer_size);
        size_t outSize_perthread = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        int prior = 0;
        unsigned int *temp_sign_arr =
            (unsigned int *)malloc(block_size * sizeof(unsigned int));
        unsigned int *temp_predict_arr =
            (unsigned int *)malloc(block_size * sizeof(unsigned int));
        int *temp_quant_arr = (int *)malloc(block_size * sizeof(int));

        if (lo < hi && outputBytes_perthread != NULL &&
            temp_sign_arr != NULL && temp_predict_arr != NULL &&
            temp_quant_arr != NULL) {
            prior = (int)(op[lo] * inver_bound);
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);

            for (size_t i = lo + 1; i < hi; i += block_size) {
                size_t current_block_size =
                    (i + block_size > hi) ? (hi - i) : block_size;
                if (current_block_size == 0) {
                    continue;
                }

                size_t compressed_block_size = worker(
                    block_pointer, op + i, inver_bound, current_block_size,
                    &prior, temp_sign_arr, temp_predict_arr, temp_quant_arr);
                block_pointer += compressed_block_size;
                outSize_perthread += compressed_block_size;
            }
        }

        outSize_perthread_arr[tid] = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (size_t i = 1; i < nbThreads; i++) {
                offsets_perthread_arr[i] =
                    offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1];
            }
            *outSize += offsets_perthread_arr[nbThreads - 1] +
                        outSize_perthread_arr[nbThreads - 1];
            memcpy(outputBytes, offsets_perthread_arr,
                   nbThreads * sizeof(size_t));
        }
#pragma omp barrier
        if (outSize_perthread != 0) {
            memcpy(real_outputBytes + offsets_perthread_arr[tid],
                   outputBytes_perthread, outSize_perthread);
        }

        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
        free(temp_quant_arr);
#pragma omp barrier
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }

    *outSize += sizeof(float);
#else
    (void)output;
    (void)oriData;
    (void)outSize;
    (void)absErrBound;
    (void)nbEle;
    (void)blockSize;
    (void)worker;
    printf("Error! OpenMP not supported!\n");
#endif
}

void szp_float_compress_blockaligned(unsigned char *output, float *oriData,
                                     size_t *outSize, float absErrBound,
                                     size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    float *op = oriData;
    unsigned char *outputBytes = output + sizeof(float);
    unsigned char *real_outputBytes = NULL;
    size_t *outSize_perthread_arr = NULL;
    size_t *offsets_perthread_arr = NULL;
    double inver_bound = 1.0 / absErrBound;
    unsigned int block_size = (unsigned int)blockSize;
    unsigned int nbThreads = 0;
    size_t threadblocksize = 0;
    size_t maxPreservedBufferSize_perthread = 0;

    floatToBytes(output, absErrBound);
    (*outSize) = 0;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t);
            outSize_perthread_arr =
                (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr =
                (size_t *)malloc(nbThreads * sizeof(size_t));
            threadblocksize = szp_float_blockaligned_threadblocksize(
                nbEle, block_size, nbThreads);
        }

        size_t i = 0;
        size_t j = 0;
        maxPreservedBufferSize_perthread =
            (sizeof(float) * nbEle + nbThreads - 1) / nbThreads +
            nbEle / nbThreads;
        unsigned char *outputBytes_perthread =
            (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;
        int tid = omp_get_thread_num();
        size_t lo = (size_t)tid * threadblocksize;
        size_t hi = (size_t)(tid + 1) * threadblocksize;
        int prior = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;

        if (tid == (int)nbThreads - 1) {
            hi = nbEle;
        }

        if (lo < hi) {
            prior = (int)(op[lo] * inver_bound);
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);
        }

        unsigned int *temp_sign_arr =
            (unsigned int *)malloc(block_size * sizeof(unsigned int));
        unsigned int *temp_predict_arr =
            (unsigned int *)malloc(block_size * sizeof(unsigned int));
        int *temp_quant_arr = (int *)malloc(block_size * sizeof(int));
        unsigned int signbytelength = 0;
        unsigned int savedbitsbytelength = 0;

        for (i = lo + 1; i < hi; i += block_size) {
            size_t current_block_size =
                (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) {
                continue;
            }

            for (j = 0; j < current_block_size; j++) {
                temp_quant_arr[j] = (int)(op[i + j] * inver_bound);
            }

            int *temp_int_predict_arr = (int *)temp_predict_arr;
            temp_int_predict_arr[0] = temp_quant_arr[0] - prior;
            for (j = 1; j < current_block_size; j++) {
                temp_int_predict_arr[j] =
                    temp_quant_arr[j] - temp_quant_arr[j - 1];
            }

            max = 0U;
            for (j = 0; j < current_block_size; j++) {
                int diff = temp_int_predict_arr[j];
                int mask = diff >> 31;
                unsigned int abs_diff = (unsigned int)((diff ^ mask) - mask);

                temp_sign_arr[j] = (unsigned int)(mask & 1);
                temp_int_predict_arr[j] = (int)abs_diff;
                if (abs_diff > max) {
                    max = abs_diff;
                }
            }

            prior = temp_quant_arr[current_block_size - 1];
            if (max == 0U) {
                *block_pointer++ = 0U;
                outSize_perthread++;
            } else {
#if defined(__GNUC__) || defined(__clang__)
                bit_count = (unsigned int)(sizeof(unsigned int) * 8U -
                                           __builtin_clz(max));
#else
                bit_count = (unsigned int)(log2f((float)max)) + 1U;
#endif
                *block_pointer++ = (unsigned char)bit_count;
                outSize_perthread++;
                signbytelength = (unsigned int)
                    Jiajun_convertUInt2Byte_fast_1b_args_MSB_SIMD(
                        temp_sign_arr, current_block_size, block_pointer);
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                savedbitsbytelength = (unsigned int)
                    Jiajun_save_fixed_length_bits_MSB(
                        temp_predict_arr, current_block_size, block_pointer,
                        bit_count);
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }

        outSize_perthread_arr[tid] = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++) {
                offsets_perthread_arr[i] =
                    offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1];
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] +
                          outSize_perthread_arr[nbThreads - 1];
            memcpy(outputBytes, offsets_perthread_arr,
                   nbThreads * sizeof(size_t));
        }
#pragma omp barrier
        memcpy(real_outputBytes + offsets_perthread_arr[tid],
               outputBytes_perthread, outSize_perthread);

        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
        free(temp_quant_arr);
#pragma omp barrier
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }

    (*outSize) += sizeof(float);
#else
    printf("Error! OpenMP not supported!\n");
#endif
}

void szp_float_compress_vecBlockaligned(unsigned char *output, float *oriData,
                                        size_t *outSize, float absErrBound,
                                        size_t nbEle, int blockSize)
{
    szp_float_openmp_blockaligned_impl(
        output, oriData, outSize, absErrBound, nbEle, blockSize,
        szp_float_vec_blockaligned_worker);
}

void szp_float_compress_vecBlockaligned_singlepass(unsigned char *output,
                                                   float *oriData,
                                                   size_t *outSize,
                                                   float absErrBound,
                                                   size_t nbEle,
                                                   int blockSize)
{
    szp_float_openmp_blockaligned_impl(
        output, oriData, outSize, absErrBound, nbEle, blockSize,
        szp_float_vec_blockaligned_singlepass_worker);
}

unsigned char *
szp_float_openmp_threadblock_randomaccess(float *oriData, size_t *outSize,
                                          float absErrBound, size_t nbEle,
                                          int blockSize)
{
#ifdef _OPENMP
    float *op = oriData;
    size_t maxPreservedBufferSize = sizeof(float) + sizeof(float) * nbEle;
    unsigned char *output = (unsigned char *)malloc(maxPreservedBufferSize);
    unsigned char *outputBytes = output + sizeof(float);
    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *real_outputBytes = NULL;
    size_t *outSize_perthread_arr = NULL;
    size_t *offsets_perthread_arr = NULL;
    unsigned int nbThreads = 0;
    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int block_size = (unsigned int)blockSize;

    floatToBytes(output, absErrBound);
    (*outSize) = 0;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = (unsigned int)omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t);
            outSize_perthread_arr =
                (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr =
                (size_t *)malloc(nbThreads * sizeof(size_t));
            maxPreservedBufferSize_perthread =
                (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
            inver_bound = 1.0 / absErrBound;
            threadblocksize = nbEle / nbThreads;
        }

        size_t i = 0;
        size_t j = 0;
        unsigned char *outputBytes_perthread =
            (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;
        int tid = omp_get_thread_num();
        size_t lo = (size_t)tid * threadblocksize;
        size_t hi = (size_t)(tid + 1) * threadblocksize;
        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        unsigned char *temp_sign_arr =
            (unsigned char *)malloc((block_size - 1) * sizeof(unsigned char));
        unsigned int *temp_predict_arr =
            (unsigned int *)malloc((block_size - 1) * sizeof(unsigned int));

        if (tid == (int)nbThreads - 1) {
            hi = nbEle;
        }

        for (i = lo; i < hi; i += block_size) {
            size_t current_block_size =
                (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) {
                continue;
            }

            max = 0;
            prior = (int)(op[i] * inver_bound);
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);

            if (current_block_size > 1) {
                for (j = 0; j < current_block_size - 1; j++) {
                    current = (int)(op[i + j + 1] * inver_bound);
                    diff = current - prior;
                    prior = current;
                    if (diff == 0) {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = 0;
                    } else {
                        if (diff < 0) {
                            temp_sign_arr[j] = 1;
                            temp_predict_arr[j] = (unsigned int)(-diff);
                        } else {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = (unsigned int)diff;
                        }
                        if (max < temp_predict_arr[j]) {
                            max = temp_predict_arr[j];
                        }
                    }
                }
            }

            if (max == 0U) {
                *block_pointer++ = 0;
                outSize_perthread++;
            } else {
                unsigned int signbytelength = 0;
                unsigned int savedbitsbytelength = 0;

                bit_count = (unsigned int)(log2f((float)max)) + 1U;
                *block_pointer++ = (unsigned char)bit_count;
                outSize_perthread++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(
                    temp_sign_arr, current_block_size - 1, block_pointer);
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                savedbitsbytelength = Jiajun_save_fixed_length_bits(
                    temp_predict_arr, current_block_size - 1, block_pointer,
                    bit_count);
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }

        outSize_perthread_arr[tid] = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++) {
                offsets_perthread_arr[i] =
                    offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1];
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] +
                          outSize_perthread_arr[nbThreads - 1];
            memcpy(outputBytes, offsets_perthread_arr,
                   nbThreads * sizeof(size_t));
        }
#pragma omp barrier
        memcpy(real_outputBytes + offsets_perthread_arr[tid],
               outputBytes_perthread, outSize_perthread);
#pragma omp barrier

        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }

    (*outSize) += sizeof(float);
    return output;
#else
    printf("Error! OpenMP not supported!\n");
    return NULL;
#endif
}

void szp_float_openmp_threadblock_randomaccess_arg(
    unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    float *op = oriData;
    unsigned char *outputBytes = output + sizeof(float);
    size_t maxPreservedBufferSize = sizeof(float) + sizeof(float) * nbEle;
    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *real_outputBytes = NULL;
    size_t *outSize_perthread_arr = NULL;
    size_t *offsets_perthread_arr = NULL;
    unsigned int nbThreads = 0;
    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int block_size = (unsigned int)blockSize;
    unsigned int new_block_size = block_size - 1U;

    floatToBytes(output, absErrBound);
    (*outSize) = 0;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = (unsigned int)omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t);
            outSize_perthread_arr =
                (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr =
                (size_t *)malloc(nbThreads * sizeof(size_t));
            maxPreservedBufferSize_perthread =
                (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
            inver_bound = 1.0 / absErrBound;
            threadblocksize = nbEle / nbThreads;
        }

        size_t i = 0;
        size_t j = 0;
        unsigned char *outputBytes_perthread =
            (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;
        int tid = omp_get_thread_num();
        size_t lo = (size_t)tid * threadblocksize;
        size_t hi = (size_t)(tid + 1) * threadblocksize;
        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        unsigned char *temp_sign_arr =
            (unsigned char *)malloc(new_block_size * sizeof(unsigned char));
        unsigned int *temp_predict_arr =
            (unsigned int *)malloc(new_block_size * sizeof(unsigned int));

        if (tid == (int)nbThreads - 1) {
            hi = nbEle;
        }

        for (i = lo; i < hi; i += block_size) {
            size_t current_block_size =
                (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) {
                continue;
            }

            max = 0;
            prior = (int)(op[i] * inver_bound);
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);

            if (current_block_size > 1) {
                for (j = 0; j < current_block_size - 1; j++) {
                    current = (int)(op[i + j + 1] * inver_bound);
                    diff = current - prior;
                    prior = current;
                    if (diff == 0) {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = 0;
                    } else {
                        if (diff < 0) {
                            temp_sign_arr[j] = 1;
                            temp_predict_arr[j] = (unsigned int)(-diff);
                        } else {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = (unsigned int)diff;
                        }
                        if (max < temp_predict_arr[j]) {
                            max = temp_predict_arr[j];
                        }
                    }
                }
            }

            if (max == 0U) {
                *block_pointer++ = 0;
                outSize_perthread++;
            } else {
                unsigned int signbytelength = 0;
                unsigned int savedbitsbytelength = 0;

                bit_count = (unsigned int)(log2f((float)max)) + 1U;
                *block_pointer++ = (unsigned char)bit_count;
                outSize_perthread++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(
                    temp_sign_arr, current_block_size - 1, block_pointer);
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                savedbitsbytelength = Jiajun_save_fixed_length_bits(
                    temp_predict_arr, current_block_size - 1, block_pointer,
                    bit_count);
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }

        outSize_perthread_arr[tid] = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++) {
                offsets_perthread_arr[i] =
                    offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1];
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] +
                          outSize_perthread_arr[nbThreads - 1];
            memcpy(outputBytes, offsets_perthread_arr,
                   nbThreads * sizeof(size_t));
        }
#pragma omp barrier
        memcpy(real_outputBytes + offsets_perthread_arr[tid],
               outputBytes_perthread, outSize_perthread);
#pragma omp barrier

        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }

    (*outSize) += sizeof(float);
#else
    printf("Error! OpenMP not supported!\n");
#endif
}

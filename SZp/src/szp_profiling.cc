/**
 *  @file szp_profiling.cc
 *  @author Jiefeng Zhou
 *  @date Oct, 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include "szp.h"
#include "szp_float.h"
#include "szpd_float.h"
#include "szp_profiling.h"
#include <assert.h>
#include <math.h>
#include "szp_TypeManager.h"
#include "szp_CompressionToolkit.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>

#ifdef _OPENMP
#include "omp.h"
#endif

using namespace szp;

struct szp_prof_data szp_float_single_thread_arg_profile(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                 size_t nbEle, int blockSize)
{
    struct szp_prof_data data = {0};
    auto start = std::chrono::high_resolution_clock::now();

    float *op = oriData;

    unsigned char* outputBytes = output + sizeof(float);
    floatToBytes(output, absErrBound);
    
    unsigned char *real_outputBytes;
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    (*outSize) = 0;

    double inver_bound = 1 / absErrBound;
    unsigned int block_size = blockSize;

    int nbThreads = 1;
    real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
    (*outSize) += nbThreads * sizeof(size_t); 
    outSize_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
    offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

    auto allocation_end = std::chrono::high_resolution_clock::now();
    data.allocation = std::chrono::duration<double>(allocation_end - start).count();

    size_t maxPreservedBufferSize_perthread = sizeof(float) * nbEle + blockSize;
    unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
    size_t outSize_perthread = 0;
    
    int tid = 0;
    size_t lo = 0;
    size_t hi = nbEle;

    int prior = 0;
    int current = 0;
    int diff = 0;
    unsigned int max = 0;
    unsigned int bit_count = 0;
    unsigned char *block_pointer = outputBytes_perthread;
    
    if (lo < hi) {
        prior = (op[lo]) * inver_bound;
        memcpy(block_pointer, &prior, sizeof(int));
        block_pointer += sizeof(unsigned int);
        outSize_perthread += sizeof(unsigned int);
    }
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int signbytelength = 0; 
    unsigned int savedbitsbytelength = 0;
    
    for (size_t i = lo + 1; i < hi; i = i + block_size)
    {
        size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
        if (current_block_size == 0) continue;

        max = 0;
        for (size_t j = 0; j < current_block_size; j++)
        {
            auto t_quant_start = std::chrono::high_resolution_clock::now();
            current = (op[i + j]) * inver_bound;
            //Quantization Ops=2:  Read op, write current

            auto t_pred_start = std::chrono::high_resolution_clock::now();
            diff = current - prior;
            prior = current;
            //Prediction Ops=4: Read current, prior; Write diff, prior

            auto t_sign_start = std::chrono::high_resolution_clock::now();
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
            auto t_sign_end = std::chrono::high_resolution_clock::now();
            //sign abs max Ops=5: Read diff, write sign, write abs, read predict_arr, read/write max

            data.t_quant += std::chrono::duration<double>(t_pred_start - t_quant_start).count();
            data.t_pred += std::chrono::duration<double>(t_sign_start - t_pred_start).count();
            data.t_sign_abs_max += std::chrono::duration<double>(t_sign_end - t_sign_start).count();
        }

        auto packing_start = std::chrono::high_resolution_clock::now();
        if (max == 0) 
        {
            block_pointer[0] = 0;
            block_pointer++;
            outSize_perthread++;
            data.ops_packing++;
        }
        else
        {
            bit_count = (int)(log2f(max)) + 1;
            block_pointer[0] = bit_count;
            data.ops_packing++;
            
            outSize_perthread++;
            block_pointer++;
            signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size, block_pointer); 
            data.ops_packing += current_block_size + signbytelength; // Reads from temp_sign_arr + writes to block_pointer
            block_pointer += signbytelength;
            outSize_perthread += signbytelength;
            
            savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size, block_pointer, bit_count);
            data.ops_packing += current_block_size + savedbitsbytelength; // Reads from temp_predict_arr + writes to block_pointer
            
            block_pointer += savedbitsbytelength;
            outSize_perthread += savedbitsbytelength;
        }
        auto packing_end = std::chrono::high_resolution_clock::now();
        data.t_packing += std::chrono::duration<double>(packing_end - packing_start).count();
    }

    outSize_perthread_arr[tid] = outSize_perthread;
    offsets_perthread_arr[0] = 0;

    (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
    memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
    
    memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
    
    free(outputBytes_perthread);
    free(temp_sign_arr);
    free(temp_predict_arr);
    
    free(outSize_perthread_arr);
    free(offsets_perthread_arr);
    
    (*outSize) += sizeof(float);

    auto end = std::chrono::high_resolution_clock::now();
    data.total = std::chrono::duration<double>(end - start).count();
    return data;
}

struct szp_prof_data szp_float_openmp_threadblock_arg_profile(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                      size_t nbEle, int blockSize)
{
    struct szp_prof_data data = {0};
#ifdef _OPENMP
    auto start = std::chrono::high_resolution_clock::now();
    float *op = oriData;
    
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
#pragma omp single
    nbThreads = omp_get_num_threads();

    std::vector<double> v_t_quant(nbThreads, 0.0), v_t_pred(nbThreads, 0.0), v_t_sign_abs_max(nbThreads, 0.0), v_t_packing(nbThreads, 0.0);
    std::vector<size_t> v_ops_packing(nbThreads, 0);

    auto allocation_end = std::chrono::high_resolution_clock::now();
    data.allocation = std::chrono::duration<double>(allocation_end - start).count();

#pragma omp parallel
    {
#pragma omp single
        {
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t); 
            outSize_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
            inver_bound = 1 / absErrBound;
            threadblocksize = nbEle / nbThreads;
        }
        size_t i = 0;
        size_t j = 0;
        
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle; // Ensure the last thread processes all remaining elements
        }

        size_t maxPreservedBufferSize_perthread = sizeof(float) * (hi - lo) + block_size;
        unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;

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
                auto t_quant_start = std::chrono::high_resolution_clock::now();
                current = (op[i + j]) * inver_bound;

                auto t_pred_start = std::chrono::high_resolution_clock::now();
                diff = current - prior;
                prior = current;

                auto t_sign_start = std::chrono::high_resolution_clock::now();
                if (diff == 0) {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = 0;
                } else {
                    if (diff < 0) {
                        temp_sign_arr[j] = 1;
                        temp_predict_arr[j] = -diff;
                    } else {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = diff;
                    }
                    if (max < temp_predict_arr[j]) max = temp_predict_arr[j];
                }
                auto t_sign_end = std::chrono::high_resolution_clock::now();

                v_t_quant[tid] += std::chrono::duration<double>(t_pred_start - t_quant_start).count();
                v_t_pred[tid] += std::chrono::duration<double>(t_sign_start - t_pred_start).count();
                v_t_sign_abs_max[tid] += std::chrono::duration<double>(t_sign_end - t_sign_start).count();
            }

            auto packing_start = std::chrono::high_resolution_clock::now();
            if (max == 0) {
                block_pointer[0] = 0;
                block_pointer++;
                outSize_perthread++;
                v_ops_packing[tid]++;
            } else {
                bit_count = (int)(log2f(max)) + 1;
                block_pointer[0] = bit_count;
                v_ops_packing[tid]++;
                
                outSize_perthread++;
                block_pointer++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size, block_pointer); 
                v_ops_packing[tid] += current_block_size + signbytelength;
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                
                savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size, block_pointer, bit_count);
                v_ops_packing[tid] += current_block_size + savedbitsbytelength;
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
            auto packing_end = std::chrono::high_resolution_clock::now();
            v_t_packing[tid] += std::chrono::duration<double>(packing_end - packing_start).count();
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
    auto end = std::chrono::high_resolution_clock::now();
    data.total = std::chrono::duration<double>(end - start).count();

    data.t_quant = std::accumulate(v_t_quant.begin(), v_t_quant.end(), 0.0);
    data.t_pred = std::accumulate(v_t_pred.begin(), v_t_pred.end(), 0.0);
    data.t_sign_abs_max = std::accumulate(v_t_sign_abs_max.begin(), v_t_sign_abs_max.end(), 0.0);
    data.t_packing = std::accumulate(v_t_packing.begin(), v_t_packing.end(), 0.0);

    data.ops_packing = std::accumulate(v_ops_packing.begin(), v_ops_packing.end(), (size_t)0);

#else
    printf("Error! OpenMP not supported!\n");
#endif
    return data;
}

struct szp_prof_data szp_float_decompress_single_thread_arg_profile(float *newData, size_t nbEle,
                                            float absErrBound, int blockSize,
                                            unsigned char *cmpBytes)
{
    struct szp_prof_data data = {0};
    auto start = std::chrono::high_resolution_clock::now();
    size_t *offsets = (size_t *)(cmpBytes + sizeof(float));
    unsigned char *rcp;
    unsigned int nbThreads = 1;
    
    rcp = cmpBytes + sizeof(float) + nbThreads * sizeof(size_t);
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
    auto allocation_end = std::chrono::high_resolution_clock::now();
    data.allocation = std::chrono::duration<double>(allocation_end - start).count();
    
    for (i = lo + 1; i < hi; i = i + block_size)
    {
        size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
        if (current_block_size == 0) continue;

        auto unpacking_start = std::chrono::high_resolution_clock::now();
        bit_count = block_pointer[0];
        block_pointer++;
        data.ops_unpacking++;
        
        if (bit_count == 0)
        {
            auto unpacking_end = std::chrono::high_resolution_clock::now();
            data.t_unpacking += std::chrono::duration<double>(unpacking_end - unpacking_start).count();

            auto dequant_recon_start = std::chrono::high_resolution_clock::now();
            ori_prior = (float)prior * absErrBound;
            //Quantization Ops=2: read prior, write ori_prior
            
            for (j = 0; j < current_block_size; j++)
            {
                memcpy(newData_perthread, &ori_prior, sizeof(float));
                newData_perthread++;
            }
            //Ops=2: read ori_prior, write to newData
            auto dequant_recon_end = std::chrono::high_resolution_clock::now();
            data.t_dequant += std::chrono::duration<double>(dequant_recon_end - dequant_recon_start).count();
        }
        else
        {
            size_t sign_bytes = (current_block_size - 1) / 8 + 1;
            convertByteArray2IntArray_fast_1b_args(current_block_size, block_pointer, sign_bytes, temp_sign_arr);
            data.ops_unpacking += sign_bytes + current_block_size; // Reads + Writes
            block_pointer += sign_bytes;

            savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size, temp_predict_arr, bit_count);
            data.ops_unpacking += savedbitsbytelength + current_block_size; // Reads + Writes
            block_pointer += savedbitsbytelength;
            auto unpacking_end = std::chrono::high_resolution_clock::now();
            data.t_unpacking += std::chrono::duration<double>(unpacking_end - unpacking_start).count();

            for (j = 0; j < current_block_size; j++)
            {
                auto t_sign_restore_start = std::chrono::high_resolution_clock::now();
                if (temp_sign_arr[j] == 0) {
                    diff = temp_predict_arr[j];
                } else {
                    diff = 0 - temp_predict_arr[j];
                }
                //Sign Resto. Ops=3 read sign, read abs, write diff

                auto t_pred_recon_start = std::chrono::high_resolution_clock::now();
                current = prior + diff;
                //Pred. Recon. Ops=3 read prior, read diff, write current

                auto t_dequant_start = std::chrono::high_resolution_clock::now();
                ori_current = (float)current * absErrBound;
                prior = current;
                memcpy(newData_perthread, &ori_current, sizeof(float));
                newData_perthread++;
                auto t_dequant_end = std::chrono::high_resolution_clock::now();
                //Dequantization Ops:5 read current, write ori_current, write prior, memcpy(read, write)

                data.t_sign_restore += std::chrono::duration<double>(t_pred_recon_start - t_sign_restore_start).count();
                data.t_pred_recon += std::chrono::duration<double>(t_dequant_start - t_pred_recon_start).count();
                data.t_dequant += std::chrono::duration<double>(t_dequant_end - t_dequant_start).count();
            }
        }
    }
    
    free(temp_predict_arr);
    free(temp_sign_arr);

    auto end = std::chrono::high_resolution_clock::now();
    data.total = std::chrono::duration<double>(end - start).count();
    return data;
}

struct szp_prof_data szp_float_decompress_openmp_threadblock_arg_profile(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
    struct szp_prof_data data = {0};
#ifdef _OPENMP
    auto start = std::chrono::high_resolution_clock::now();
    size_t *offsets = (size_t *)(cmpBytes + sizeof(float));
    unsigned char *rcp;
    unsigned int nbThreads = 0;
    
    size_t threadblocksize = 0;
    size_t block_size = blockSize;

#pragma omp parallel
#pragma omp single
    nbThreads = omp_get_num_threads();

    std::vector<double> v_t_unpacking(nbThreads, 0.0), v_t_sign_restore(nbThreads, 0.0), v_t_pred_recon(nbThreads, 0.0), v_t_dequant(nbThreads, 0.0);

    auto allocation_end = std::chrono::high_resolution_clock::now();
    data.allocation = std::chrono::duration<double>(allocation_end - start).count();
    std::vector<size_t> v_ops_unpacking(nbThreads, 0);

#pragma omp parallel
    {
#pragma omp single
        {
            rcp = cmpBytes + sizeof(float) + nbThreads * sizeof(size_t);
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

            auto unpacking_start = std::chrono::high_resolution_clock::now();
            bit_count = block_pointer[0];
            block_pointer++;
            v_ops_unpacking[tid]++;
            
            if (bit_count == 0)
            {
                auto unpacking_end = std::chrono::high_resolution_clock::now();
                v_t_unpacking[tid] += std::chrono::duration<double>(unpacking_end - unpacking_start).count();

                auto dequant_recon_start = std::chrono::high_resolution_clock::now();
                ori_prior = (float)prior * absErrBound;
                
                for (j = 0; j < current_block_size; j++)
                {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                }
                auto dequant_recon_end = std::chrono::high_resolution_clock::now();
                v_t_dequant[tid] += std::chrono::duration<double>(dequant_recon_end - dequant_recon_start).count();
            }
            else
            {
                size_t sign_bytes = (current_block_size - 1) / 8 + 1;
                convertByteArray2IntArray_fast_1b_args(current_block_size, block_pointer, sign_bytes, temp_sign_arr);
                v_ops_unpacking[tid] += sign_bytes + current_block_size;
                block_pointer += sign_bytes;

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size, temp_predict_arr, bit_count);
                v_ops_unpacking[tid] += savedbitsbytelength + current_block_size;
                block_pointer += savedbitsbytelength;
                auto unpacking_end = std::chrono::high_resolution_clock::now();
                v_t_unpacking[tid] += std::chrono::duration<double>(unpacking_end - unpacking_start).count();

                for (j = 0; j < current_block_size; j++)
                {
                    auto t_sign_restore_start = std::chrono::high_resolution_clock::now();
                    if (temp_sign_arr[j] == 0) {
                        diff = temp_predict_arr[j];
                    } else {
                        diff = 0 - temp_predict_arr[j];
                    }

                    auto t_pred_recon_start = std::chrono::high_resolution_clock::now();
                    current = prior + diff;

                    auto t_dequant_start = std::chrono::high_resolution_clock::now();
                    ori_current = (float)current * absErrBound;
                    prior = current;
                    memcpy(newData_perthread, &ori_current, sizeof(float));
                    newData_perthread++;
                    auto t_dequant_end = std::chrono::high_resolution_clock::now();

                    v_t_sign_restore[tid] += std::chrono::duration<double>(t_pred_recon_start - t_sign_restore_start).count();
                    v_t_pred_recon[tid] += std::chrono::duration<double>(t_dequant_start - t_pred_recon_start).count();
                    v_t_dequant[tid] += std::chrono::duration<double>(t_dequant_end - t_dequant_start).count();
                }
            }
        }
        free(temp_predict_arr);
        free(temp_sign_arr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    data.total = std::chrono::duration<double>(end - start).count();

    data.t_unpacking = std::accumulate(v_t_unpacking.begin(), v_t_unpacking.end(), 0.0);
    data.t_sign_restore = std::accumulate(v_t_sign_restore.begin(), v_t_sign_restore.end(), 0.0);
    data.t_pred_recon = std::accumulate(v_t_pred_recon.begin(), v_t_pred_recon.end(), 0.0);
    data.t_dequant = std::accumulate(v_t_dequant.begin(), v_t_dequant.end(), 0.0);

    data.ops_unpacking = std::accumulate(v_ops_unpacking.begin(), v_ops_unpacking.end(), (size_t)0);


#else
    printf("Error! OpenMP not supported!\n");
#endif
    return data;
}


static size_t szp_float_block_compiler_buffer_profile(
    unsigned char *__restrict__ block_pointer, float *__restrict__ op,
    double inver_bound, size_t current_block_size, int *__restrict__ prior,
    unsigned char *__restrict__ temp_sign_arr,
    unsigned int *__restrict__ temp_predict_arr,
    int *__restrict__ temp_quant_arr,
    struct szp_prof_data* prof_data)
{
    unsigned char *start_block_pointer = block_pointer;
    int *temp_int_predict_arr = reinterpret_cast<int *>(temp_predict_arr);

    // 1. Quantize
    auto t_quant_start = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < current_block_size; j++) {
        temp_quant_arr[j] = (int)(op[j] * inver_bound);
    }
    auto t_quant_end = std::chrono::high_resolution_clock::now();
    prof_data->t_quant += std::chrono::duration<double>(t_quant_end - t_quant_start).count();
    
    // 2. Compute differences
    auto t_pred_start = std::chrono::high_resolution_clock::now();
    temp_int_predict_arr[0] = temp_quant_arr[0] - *prior;
    for (size_t j = 1; j < current_block_size; j++) {
        temp_int_predict_arr[j] = temp_quant_arr[j] - temp_quant_arr[j - 1];
    }
    auto t_pred_end = std::chrono::high_resolution_clock::now();
    prof_data->t_pred += std::chrono::duration<double>(t_pred_end - t_pred_start).count();

    // 3. Compute sign, abs, and max
    auto t_sign_start = std::chrono::high_resolution_clock::now();
    int max = 0;
    for (size_t j = 0; j < current_block_size; j++) {
        int diff = temp_int_predict_arr[j];
        temp_sign_arr[j] = diff < 0 ? 1 : 0;
        diff = abs(diff);
        temp_int_predict_arr[j] = diff;
        max = (diff > max) ? diff : max;
    }
    auto t_sign_end = std::chrono::high_resolution_clock::now();
    prof_data->t_sign_abs_max += std::chrono::duration<double>(t_sign_end - t_sign_start).count();

    // Update prior
    *prior = temp_quant_arr[current_block_size - 1];

    // 4. Packing
    auto packing_start = std::chrono::high_resolution_clock::now();
    if (max == 0) {
        block_pointer[0] = 0;
        prof_data->ops_packing++;
        block_pointer++;
    } else {
        unsigned int bit_count = 0;
#if defined(__GNUC__) || defined(__clang__)
        bit_count = sizeof(unsigned int) * 8 - __builtin_clz(max);
#else
        bit_count = (unsigned int)(log2f(max)) + 1;
#endif
        block_pointer[0] = bit_count;
        prof_data->ops_packing++;
        block_pointer++;

        unsigned int signbytelength = convertIntArray2ByteArray_fast_1b_args(
            temp_sign_arr, current_block_size, block_pointer);
        prof_data->ops_packing += current_block_size + signbytelength;
        block_pointer += signbytelength;

        unsigned int savedbitsbytelength = Jiajun_save_fixed_length_bits(
            temp_predict_arr, current_block_size, block_pointer, bit_count);
        prof_data->ops_packing += current_block_size + savedbitsbytelength;
        block_pointer += savedbitsbytelength;
    }
    auto packing_end = std::chrono::high_resolution_clock::now();
    prof_data->t_packing += std::chrono::duration<double>(packing_end - packing_start).count();

    return block_pointer - start_block_pointer;
}

/**
 * @brief Single-threaded profiling function.
 * This function profiles the compression algorithm by processing the data in sequential blocks.
 */
struct szp_prof_data szp_float_single_thread_arg_buffer_profile(
    unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize)
{
    struct szp_prof_data data = {0};
    auto start = std::chrono::high_resolution_clock::now();

    float *op = oriData;
    *outSize = 0;

    floatToBytes(output, absErrBound);
    *outSize += sizeof(float);

    double inver_bound = 1 / absErrBound;
    unsigned int block_size = static_cast<unsigned int>(blockSize);

    unsigned char *block_pointer = output + sizeof(float);

    if (nbEle == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        data.total = std::chrono::duration<double>(end - start).count();
        return data;
    }
    
    // Allocate temp buffers
    unsigned char *temp_sign_arr = (unsigned char *)malloc(block_size * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc(block_size * sizeof(unsigned int));
    int *temp_quant_arr = (int *)malloc(block_size * sizeof(int));
    
    auto allocation_end = std::chrono::high_resolution_clock::now();
    data.allocation = std::chrono::duration<double>(allocation_end - start).count();

    // Initialize and write the first value (prior)
    int prior = (op[0]) * inver_bound;
    memcpy(block_pointer, &prior, sizeof(int));
    block_pointer += sizeof(int);

    // Process subsequent data in blocks
    for (size_t i = 1; i < nbEle; i = i + block_size) {
        size_t current_block_size =
            (i + block_size > nbEle) ? (nbEle - i) : block_size;
        if (current_block_size == 0)
            continue;

        size_t compressed_block_size = szp_float_block_compiler_buffer_profile(
            block_pointer, op + i, inver_bound, current_block_size, &prior,
            temp_sign_arr, temp_predict_arr, temp_quant_arr, &data
        );
        block_pointer += compressed_block_size;
    }
    
    *outSize = (block_pointer - output);

    // Free temp buffers
    free(temp_sign_arr);
    free(temp_predict_arr);
    free(temp_quant_arr);

    auto end = std::chrono::high_resolution_clock::now();
    data.total = std::chrono::duration<double>(end - start).count();
    return data;
}

/**
 * @brief OpenMP-parallelized profiling function.
 * This function profiles the compression algorithm by dividing the data among multiple
 * threads, each processing its chunk in blocks.
 */
struct szp_prof_data szp_float_openmp_threadblock_arg_buffer_profile(
    unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize)
{
    struct szp_prof_data data = {0};
#ifdef _OPENMP
    auto start = std::chrono::high_resolution_clock::now();
    
    float *op = oriData;
    unsigned char *real_outputBytes;
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    unsigned char* outputBytes = output + sizeof(float);
    floatToBytes(output, absErrBound);

    (*outSize) = 0;

    unsigned int nbThreads = 0;
    double inver_bound = 0;
    size_t threadblocksize = 0;
    unsigned int block_size = blockSize;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            nbThreads = omp_get_num_threads();
        }
    }

    std::vector<szp_prof_data> thread_data(nbThreads);

    auto allocation_end = std::chrono::high_resolution_clock::now();
    data.allocation = std::chrono::duration<double>(allocation_end - start).count();

#pragma omp parallel
    {
#pragma omp single
        {
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t);
            outSize_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

            inver_bound = 1 / absErrBound;
            threadblocksize = (nbEle + nbThreads - 1) / nbThreads;
        }

        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (hi > nbEle) {
            hi = nbEle;
        }
        
        size_t maxPreservedBufferSize_perthread = sizeof(float) * (hi - lo) + block_size;
        unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        
        if (lo < hi) {
            unsigned char *temp_sign_arr = (unsigned char *)malloc(block_size * sizeof(unsigned char));
            unsigned int *temp_predict_arr = (unsigned int *)malloc(block_size * sizeof(unsigned int));
            int *temp_quant_arr = (int *)malloc(block_size * sizeof(int));

            int prior = (op[lo]) * inver_bound;
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(int);
            outSize_perthread += sizeof(int);

            for (size_t i = lo + 1; i < hi; i = i + block_size)
            {
                size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
                if (current_block_size == 0) continue;

                size_t compressed_block_size = szp_float_block_compiler_buffer_profile(
                    block_pointer, op + i, inver_bound, current_block_size, &prior,
                    temp_sign_arr, temp_predict_arr, temp_quant_arr, &thread_data[tid]
                );
                
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
            for (size_t i = 1; i < nbThreads; i++)
            {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1];
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
        }
#pragma omp barrier
        if (lo < hi) {
            memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
        }
        free(outputBytes_perthread);

#pragma omp barrier
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }
    
    // Accumulate results from all threads
    for (unsigned int i = 0; i < nbThreads; i++) {
        data.t_quant += thread_data[i].t_quant;
        data.t_pred += thread_data[i].t_pred;
        data.t_sign_abs_max += thread_data[i].t_sign_abs_max;
        data.t_packing += thread_data[i].t_packing;
        data.ops_packing += thread_data[i].ops_packing;
    }

    (*outSize) += sizeof(float);
    auto end = std::chrono::high_resolution_clock::now();
    data.total = std::chrono::duration<double>(end - start).count();

#else
    printf("Error! OpenMP not supported!\n");
    // If OpenMP is not supported, fall back to the single-threaded profiler.
    data = szp_float_single_thread_arg_profile(output, oriData, outSize, absErrBound, nbEle, blockSize);
#endif
    return data;
}

static size_t szp_float_decompress_block_compiler_buffer_profile(
    unsigned char *__restrict__ block_pointer,
    float *__restrict__ newData_perthread, size_t current_block_size,
    float absErrBound, int *__restrict__ prior,
    unsigned char *__restrict__ temp_sign_arr,
    unsigned int *__restrict__ temp_predict_arr,
    struct szp_prof_data* prof_data)
{
    unsigned char *start_block_pointer = block_pointer;

    // 1. Unpacking
    auto unpacking_start = std::chrono::high_resolution_clock::now();
    unsigned int bit_count = block_pointer[0];
    prof_data->ops_unpacking++;
    block_pointer++;

    if (bit_count == 0) {
        auto unpacking_end = std::chrono::high_resolution_clock::now();
        prof_data->t_unpacking += std::chrono::duration<double>(unpacking_end - unpacking_start).count();
        
        // 4. Dequantization (special case for all-zero block)
        auto dequant_start = std::chrono::high_resolution_clock::now();
        float ori_prior = (float)(*prior) * absErrBound;
        for (size_t j = 0; j < current_block_size; j++) {
            newData_perthread[j] = ori_prior;
        }
        auto dequant_end = std::chrono::high_resolution_clock::now();
        prof_data->t_dequant += std::chrono::duration<double>(dequant_end - dequant_start).count();

    } else {
        size_t sign_byte_len = (current_block_size + 7) / 8;
        convertByteArray2IntArray_fast_1b_args(
            current_block_size, block_pointer,
            sign_byte_len, temp_sign_arr);
        prof_data->ops_unpacking += sign_byte_len + current_block_size;
        block_pointer += sign_byte_len;

        unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(
            block_pointer, current_block_size,
            temp_predict_arr, bit_count);
        prof_data->ops_unpacking += savedbitsbytelength + current_block_size;
        block_pointer += savedbitsbytelength;
        auto unpacking_end = std::chrono::high_resolution_clock::now();
        prof_data->t_unpacking += std::chrono::duration<double>(unpacking_end - unpacking_start).count();

        int *temp_int_predict_arr = reinterpret_cast<int *>(temp_predict_arr);

        // 2. Sign Restoration
        auto sign_restore_start = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < current_block_size; j++) {
            if (temp_sign_arr[j] != 0) {
                temp_int_predict_arr[j] = -static_cast<int>(temp_predict_arr[j]);
            }
        }
        auto sign_restore_end = std::chrono::high_resolution_clock::now();
        prof_data->t_sign_restore += std::chrono::duration<double>(sign_restore_end - sign_restore_start).count();

        // 3. Prediction Reconstruction (Prefix sum)
        auto pred_recon_start = std::chrono::high_resolution_clock::now();
        temp_int_predict_arr[0] += *prior;
        for (size_t j = 1; j < current_block_size; j++) {
            temp_int_predict_arr[j] += temp_int_predict_arr[j - 1];
        }
        *prior = temp_int_predict_arr[current_block_size - 1];
        auto pred_recon_end = std::chrono::high_resolution_clock::now();
        prof_data->t_pred_recon += std::chrono::duration<double>(pred_recon_end - pred_recon_start).count();

        // 4. Dequantization
        auto dequant_start = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < current_block_size; j++) {
            newData_perthread[j] =
                static_cast<float>(temp_int_predict_arr[j]) * absErrBound;
        }
        auto dequant_end = std::chrono::high_resolution_clock::now();
        prof_data->t_dequant += std::chrono::duration<double>(dequant_end - dequant_start).count();
    }
    return block_pointer - start_block_pointer;
}


// --- Main Profiling Functions ---

struct szp_prof_data szp_float_decompress_single_thread_arg_buffer_profile(
    float *newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes)
{
    struct szp_prof_data data = {0};
    auto start = std::chrono::high_resolution_clock::now();

    unsigned char *block_pointer = cmpBytes + sizeof(float);
    size_t block_size = blockSize;

    if (nbEle == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        data.total = std::chrono::duration<double>(end - start).count();
        return data;
    }

    // Allocate temporary buffers
    unsigned char *temp_sign_arr = (unsigned char *)malloc(block_size * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc(block_size * sizeof(unsigned int));
    auto allocation_end = std::chrono::high_resolution_clock::now();
    data.allocation = std::chrono::duration<double>(allocation_end - start).count();
    
    // Initialize prior and decompress the first element
    int prior = 0;
    memcpy(&prior, block_pointer, sizeof(int));
    block_pointer += sizeof(int);
    newData[0] = (float)prior * absErrBound;

    float *newData_perthread = newData + 1;

    // Process subsequent data in blocks
    for (size_t i = 1; i < nbEle; i = i + block_size) {
        size_t current_block_size =
            (i + block_size > nbEle) ? (nbEle - i) : block_size;
        if (current_block_size == 0)
            continue;

        size_t consumed_bytes = szp_float_decompress_block_compiler_buffer_profile(
            block_pointer, newData_perthread, current_block_size, absErrBound, &prior,
            temp_sign_arr, temp_predict_arr, &data
        );

        block_pointer += consumed_bytes;
        newData_perthread += current_block_size;
    }
    
    free(temp_predict_arr);
    free(temp_sign_arr);

    auto end = std::chrono::high_resolution_clock::now();
    data.total = std::chrono::duration<double>(end - start).count();
    return data;
}

struct szp_prof_data szp_float_decompress_openmp_threadblock_arg_buffer_profile(
    float *newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes)
{
    struct szp_prof_data data = {0};
#ifdef _OPENMP
    auto start = std::chrono::high_resolution_clock::now();
    size_t *offsets = (size_t *)(cmpBytes + sizeof(float));
    unsigned char *rcp;
    unsigned int nbThreads = 0;
    size_t threadblocksize = 0;
    size_t block_size = blockSize;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            nbThreads = omp_get_num_threads();
        }
    }
    
    std::vector<szp_prof_data> thread_data(nbThreads);

    auto allocation_end = std::chrono::high_resolution_clock::now();
    data.allocation = std::chrono::duration<double>(allocation_end - start).count();

#pragma omp parallel
    {
#pragma omp single
        {
            rcp = cmpBytes + sizeof(float) + nbThreads * sizeof(size_t);
            threadblocksize = (nbEle + nbThreads - 1) / nbThreads;
        }

        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (hi > nbEle) {
            hi = nbEle;
        }

        if (lo < hi) {
            float *newData_perthread = newData + lo;
            unsigned char *block_pointer = rcp + offsets[tid];

            int prior = 0;
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(int);
            *newData_perthread = (float)prior * absErrBound;
            newData_perthread++;

            unsigned char *temp_sign_arr = (unsigned char *)malloc(block_size * sizeof(unsigned char));
            unsigned int *temp_predict_arr = (unsigned int *)malloc(block_size * sizeof(unsigned int));

            for (size_t i = lo + 1; i < hi; i = i + block_size)
            {
                size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
                if (current_block_size == 0) continue;

                size_t consumed_bytes = szp_float_decompress_block_compiler_buffer_profile(
                    block_pointer, newData_perthread, current_block_size, absErrBound, &prior,
                    temp_sign_arr, temp_predict_arr, &thread_data[tid]
                );

                block_pointer += consumed_bytes;
                newData_perthread += current_block_size;
            }

            free(temp_predict_arr);
            free(temp_sign_arr);
        }
    }
    
    // Accumulate results from all threads
    for (unsigned int i = 0; i < nbThreads; i++) {
        data.t_unpacking += thread_data[i].t_unpacking;
        data.t_sign_restore += thread_data[i].t_sign_restore;
        data.t_pred_recon += thread_data[i].t_pred_recon;
        data.t_dequant += thread_data[i].t_dequant;
        data.ops_unpacking += thread_data[i].ops_unpacking;
    }

    auto end = std::chrono::high_resolution_clock::now();
    data.total = std::chrono::duration<double>(end - start).count();

#else
    printf("Error! OpenMP not supported!\n");
    // If OpenMP is not supported, fall back to the single-threaded profiler.
    data = szp_float_decompress_single_thread_arg_profile(newData, nbEle, absErrBound, blockSize, cmpBytes);
#endif
    return data;
}
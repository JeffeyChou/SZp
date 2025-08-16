/**
 *  @file compress_3d.cc
 *  @author Jiajun Huang, Sheng Di
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "szp.h"
#include <sys/time.h>
#ifdef _OPENMP
#include "omp.h"
#endif

struct timeval startTime;
struct timeval endTime;   /* Start and end times */
struct timeval costStart; /*only used for recording the cost*/
double totalCost = 0;

void cost_start()
{
    totalCost = 0;
    gettimeofday(&costStart, NULL);
}

void cost_end()
{
    double elapsed;
    struct timeval costEnd;
    gettimeofday(&costEnd, NULL);
    elapsed = ((costEnd.tv_sec * 1000000 + costEnd.tv_usec) - (costStart.tv_sec * 1000000 + costStart.tv_usec)) / 1000000.0;
    totalCost += elapsed;
}

int main(int argc, char *argv[])
{
  char oriFilePath[640], outputFilePath[645], decFilePath[645];
  if (argc < 8)
  {
    printf("Usage: compress_3d [srcFilePath] [nx] [ny] [nz] [block size] [err bound] [data_type] [access_type]\n");
    printf("Example: compress_3d temperature.f32 512 512 512 64 1E-3 float random\n");
    exit(0);
  }

  sprintf(oriFilePath, "%s", argv[1]);
  size_t nx = atoi(argv[2]);
  size_t ny = atoi(argv[3]);
  size_t nz = atoi(argv[4]);
  int blockSize = atoi(argv[5]);
  double errBound = atof(argv[6]);
  char* dataType = argv[7];
  char* accessType = argv[8];
  int sz_dataType = SZ_FLOAT;
  int accessMode = SZP_RANDOMACCESS;

  if (strcmp(dataType, "float") != 0 && strcmp(dataType, "double") != 0)
  {
    printf("Error: data type must be 'float' or 'double'!\n");
    exit(0);
  }
  if (strcmp(dataType, "double") == 0)
  {
    sz_dataType = SZ_DOUBLE;
  }

  if (strcmp(accessType, "random") == 0)
  {
    accessMode = SZP_RANDOMACCESS;
  }
  else if (strcmp(accessType, "nonrandom") == 0)
  {
    accessMode = SZP_NONRANDOMACCESS;
  }
  else
  {
    printf("Error: access type must be 'random' or 'nonrandom'!\n");
    exit(0);
  }

  sprintf(outputFilePath, "%s.szp", oriFilePath);
  sprintf(decFilePath, "%s.szp.out", oriFilePath);

  int status = 0;
  size_t nbEle;
  void *data;
  if (sz_dataType == SZ_FLOAT)
  {
    data = szp_readFloatData(oriFilePath, &nbEle, &status);
  }
  else // SZ_DOUBLE
  {
    data = szp_readDoubleData(oriFilePath, &nbEle, &status);
  }
  if (status != SZ_SCES)
  {
    printf("Error: data file %s cannot be read!\n", oriFilePath);
    exit(0);
  }

  if (nbEle != nx * ny * nz) {
      printf("Error: data size mismatch, expected %zu, but got %zu\n", nx * ny * nz, nbEle);
      exit(0);
  }

  // Compression
  size_t outSize;
  cost_start();
  unsigned char *bytes = szp_compress(accessMode, sz_dataType, data, &outSize, ABS, errBound, 0, nbEle, blockSize);
  cost_end();
  printf("\nCompression timecost=%f\n", totalCost);
  printf("compression size = %zu, CR = %f, writing to %s\n", outSize, 1.0f * nbEle * ((sz_dataType == SZ_FLOAT) ? sizeof(float) : sizeof(double))/ outSize, outputFilePath);
  szp_writeByteData(bytes, outSize, outputFilePath, &status);
  if (status != SZ_SCES)
  {
    printf("Error: data file %s cannot be written!\n", outputFilePath);
    exit(0);
  }
  printf("Compression finished.\n");

  // Decompression
  cost_start();
  void *dec_data = szp_decompress(accessMode, sz_dataType, bytes, outSize, nbEle, blockSize);
  cost_end();
  printf("\nDecompression timecost=%f\n", totalCost);
  
  szp_writeFloatData_inBytes((float*)dec_data, nbEle, decFilePath, &status);
   if (status != SZ_SCES)
    {
        printf("Error: %s cannot be written!\n", decFilePath);
        exit(0);
    }

  // Verification
  if (sz_dataType == SZ_FLOAT)
    {
        float* ori_data_f = (float*)data;
        float* data_f = (float*)dec_data;
        size_t i = 0;
        double Max = 0, Min = 0, diffMax = 0; // Use double for precision in calculations

        if (nbEle > 0) {
            Max = ori_data_f[0];
            Min = ori_data_f[0];
            diffMax = fabs(data_f[0] - ori_data_f[0]);
        }

        double sum1 = 0, sum2 = 0;
        for (i = 0; i < nbEle; i++)
        {
            sum1 += ori_data_f[i];
            sum2 += data_f[i];
        }
        double mean1 = (nbEle == 0) ? 0 : sum1 / nbEle;
        double mean2 = (nbEle == 0) ? 0 : sum2 / nbEle;

        double sum3 = 0, sum4 = 0;
        double sum = 0, prodSum = 0;
        double maxpw_relerr = 0;
        for (i = 0; i < nbEle; i++)
        {
            if (Max < ori_data_f[i]) Max = ori_data_f[i];
            if (Min > ori_data_f[i]) Min = ori_data_f[i];

            double err = fabs(data_f[i] - ori_data_f[i]);
            if (ori_data_f[i] != 0)
            {
                double relerr = err / fabs(ori_data_f[i]);
                if (maxpw_relerr < relerr)
                    maxpw_relerr = relerr;
            }
            if (diffMax < err) diffMax = err;

            prodSum += (ori_data_f[i] - mean1) * (data_f[i] - mean2);
            sum3 += (ori_data_f[i] - mean1) * (ori_data_f[i] - mean1);
            sum4 += (data_f[i] - mean2) * (data_f[i] - mean2);
            sum += err * err;
        }

        double std1 = (nbEle == 0) ? 0 : sqrt(sum3 / nbEle);
        double std2 = (nbEle == 0) ? 0 : sqrt(sum4 / nbEle);
        double ee = (nbEle == 0) ? 0 : prodSum / nbEle;
        double acEff = (std1 * std2 == 0) ? 0 : ee / (std1 * std2);
        double mse = (nbEle == 0) ? 0 : sum / nbEle;
        double range = Max - Min;
        double psnr = (mse == 0) ? 999 : 20 * log10(range) - 10 * log10(mse);
        double nrmse = (range == 0) ? 0 : sqrt(mse) / range;
        double compressionRatio = 1.0 * nbEle * sizeof(float) / outSize;

        printf("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
        printf("Max absolute error = %.10f\n", diffMax);
        printf("Max relative error = %f\n", (range == 0) ? 0 : diffMax / range);
        printf("Max pw relative error = %f\n", maxpw_relerr);
        printf("PSNR = %f, NRMSE= %.20G\n", psnr, nrmse);
        printf("acEff=%f\n", acEff);
        printf("compressionRatio = %f\n", compressionRatio);
    }
    else // SZ_DOUBLE
    {
        double* ori_data_d = (double*)data;
        double* data_d = (double*)dec_data;
        size_t i = 0;
        double Max = 0, Min = 0, diffMax = 0;
        
        if (nbEle > 0) {
            Max = ori_data_d[0];
            Min = ori_data_d[0];
            diffMax = fabs(data_d[0] - ori_data_d[0]);
        }

        double sum1 = 0, sum2 = 0;
        for (i = 0; i < nbEle; i++)
        {
            sum1 += ori_data_d[i];
            sum2 += data_d[i];
        }
        double mean1 = (nbEle == 0) ? 0 : sum1 / nbEle;
        double mean2 = (nbEle == 0) ? 0 : sum2 / nbEle;

        double sum3 = 0, sum4 = 0;
        double sum = 0, prodSum = 0;
        double maxpw_relerr = 0;
        for (i = 0; i < nbEle; i++)
        {
            if (Max < ori_data_d[i]) Max = ori_data_d[i];
            if (Min > ori_data_d[i]) Min = ori_data_d[i];
            
            double err = fabs(data_d[i] - ori_data_d[i]);
            if (ori_data_d[i] != 0)
            {
                double relerr = err / fabs(ori_data_d[i]);
                if (maxpw_relerr < relerr)
                    maxpw_relerr = relerr;
            }
            if (diffMax < err) diffMax = err;

            prodSum += (ori_data_d[i] - mean1) * (data_d[i] - mean2);
            sum3 += (ori_data_d[i] - mean1) * (ori_data_d[i] - mean1);
            sum4 += (data_d[i] - mean2) * (data_d[i] - mean2);
            sum += err * err;
        }

        double std1 = (nbEle == 0) ? 0 : sqrt(sum3 / nbEle);
        double std2 = (nbEle == 0) ? 0 : sqrt(sum4 / nbEle);
        double ee = (nbEle == 0) ? 0 : prodSum / nbEle;
        double acEff = (std1 * std2 == 0) ? 0 : ee / (std1 * std2);
        double mse = (nbEle == 0) ? 0 : sum / nbEle;
        double range = Max - Min;
        double psnr = (mse == 0) ? 999 : 20 * log10(range) - 10 * log10(mse);
        double nrmse = (range == 0) ? 0 : sqrt(mse) / range;
        double compressionRatio = 1.0 * nbEle * sizeof(double) / outSize;

        printf("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
        printf("Max absolute error = %.10f\n", diffMax);
        printf("Max relative error = %f\n", (range == 0) ? 0 : diffMax / range);
        printf("Max pw relative error = %f\n", maxpw_relerr);
        printf("PSNR = %f, NRMSE= %.20G\n", psnr, nrmse);
        printf("acEff=%f\n", acEff);
        printf("compressionRatio = %f\n", compressionRatio);
    }


  printf("done\n");
  free(bytes);
  free(data);
  free(dec_data);

  return 0;
}

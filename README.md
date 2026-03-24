# SZp (Also known as fZ-light)

* Developers: Jiajun Huang(kernels, entries, and examples), Sheng Di (utility)
* Email: jiajunhuang@usf.edu

This is the official repository of SZp, an extreme-fast error-bounded lossy compressor. It is a CPU compressor (supporting OpenMP). 
The design and optimizations of SZp are published under the name fZ-light in SC '24.

## Installation
Configure and build the SZp:
```bash
# Decompress the downloaded files
# Change to the SZp directory and set up the installation directory:
cd SZp
mkdir install

# Run the configuration script:
./configure --prefix=$(pwd)/install --enable-openmp

#Run the configuration script with additional flags that enables vectorization:
./configure --prefix=$(pwd)/install --enable-openmp --enable-vectorization --enable-avx2

# Compile the SZp using multiple threads:
make -j

# Install the compiled SZp:
make install

```
### Configuration Options
The following options are available for configuring the SZp build:
--prefix=DIR: install in DIR [default=/usr/local].
--enable-openmp: enables OpenMP support (default disabled).
--disable-openmp: disables OpenMP support.
--enable-vectorization: enables vectorization with -ftree-vectorize -funroll-loops for both C and C++.(default enabled)
--disable-vectorization: disables vectorization with -fno-tree-vectorize for both C and C++.
--enable-vec-report: enables vectorization report with -fopt-info-vec for both C and C++.(default disabled)
--disable-vec-report: disables vectorization report with -fno-opt-info-vec for both C and C++.
--enable-avx2: enables AVX2 optimizations with -mavx2 for both C and C++.(default disabled)

## Run SZp
```bash
export OMP_NUM_THREADS=$NUMTHREADS
```

## Migrated ZCCL Compressor Methods

SZp now includes the ZCCL-derived float compressor variants below, together
with the matching MSB-oriented bit-packing path used by the vectorized
implementations:

- `szp_float_compress_blockaligned`
- `szp_float_compress_vecBlockaligned`
- `szp_float_compress_vecBlockaligned_singlepass`

The corresponding decompression entry points are also available in
`SZp/SZp/src/szpd_float.cc`, so the migrated wire formats can be benchmarked
and verified directly inside SZp.

## Migrated Benchmarking

`SZp/example/test_profiling` provides a `--migrated` benchmarking mode for the
three migrated methods. It reports:

- Compression throughput: `avg`, `median`, `p95`, `std` in `GB/s`
- Compression ratio: `CR` summary plus compressed size
- Decompression throughput: `avg`, `median`, `p95`, `std` in `GB/s`
- Quality metrics: `maxAbsErr`, `maxRelErr`, `PSNR`

Example:

```bash
make -C SZp/SZp libszp.la -j4
make -C SZp/example test_profiling

OMP_NUM_THREADS=8 ./SZp/example/test_profiling --migrated \
  /mnt/c/Users/Jiefeng.Zhou/Documents/VSCODE/HPC/data/aramco-snapshot-1420.f32 \
  1e-4 128 -w 0 -r 1
```

This mode is intended for checking whether the migrated SZp implementations
match the original ZCCL compressor behavior on the same dataset and error
bound.

## Citation

If you find SZp useful in your research or applications, we kindly invite you to cite our paper. Your support helps advance the field and acknowledges the contributions of our work. Thank you!
- **[SC '24]** hZCCL: Accelerating Collective Communication with Co-Designed Homomorphic Compression
    ```bibtex
    @inproceedings{huang2024hZCCL,
        title={hZCCL: Accelerating Collective Communication with Co-Designed Homomorphic Compression},
        author = {Huang, Jiajun and Di, Sheng and Yu, Xiaodong and Zhai, Yujia and Liu, Jinyang and Jian, Zizhe and Liang, Xin and Zhao, Kai and Lu, Xiaoyi and Chen, Zizhong and Cappello, Franck and Guo, Yanfei and Thakur, Rajeev},
        booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis},
        year = {2024}
    }
    ```

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

## Run SZp
```bash
export OMP_NUM_THREADS=$NUMTHREADS
```

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

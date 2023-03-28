source /opt/rh/devtoolset-9/enable
source /opt/intel/mkl/bin/mklvars.sh intel64
export LD_LIBRARY_PATH='/home2/xiaolany/libxsmm/lib':$LD_LIBRARY_PATH
make
taskset -c 0-27 ./benchmark_nopack

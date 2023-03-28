source /opt/rh/devtoolset-9/enable
source /opt/intel/mkl/bin/mklvars.sh intel64
export LD_LIBRARY_PATH='/home2/xiaolany/libxsmm/lib':$LD_LIBRARY_PATH
make
export LD_PRELOAD="/home2/xiaolany/anaconda3/envs/pt/lib/libjemalloc.so:/home2/xiaolany/anaconda3/lib/libiomp5.so"
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=28
numactl -C 0-27 -m 0 ./benchmark_nopack


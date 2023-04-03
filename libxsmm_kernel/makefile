CXX = g++
GCCVERSIONGTEQ7 := $(shell expr `g++ -dumpversion | cut -f1 -d.` \>= 7)
# removed -freorder-blocks-algorithm=stc -freorder-blocks-and-partition
# removed -fsched-spec -fschedule-insns -fschedule-insns2 
# removed -ftree-pre 
O2_REMOVE_REORDER = -fthread-jumps -falign-functions \
           -falign-jumps -falign-loops  -falign-labels -fcaller-saves \
           -fcrossjumping -fcse-follow-jumps  -fcse-skip-blocks \
           -fdelete-null-pointer-checks -fdevirtualize -fdevirtualize-speculatively \
           -fexpensive-optimizations -fgcse  -fgcse-lm -fhoist-adjacent-loads \
           -finline-small-functions -findirect-inlining -fipa-cp -fipa-bit-cp \
           -fipa-vrp -fipa-sra -fipa-icf -fisolate-erroneous-paths-dereference \
           -flra-remat -foptimize-sibling-calls -foptimize-strlen \
           -fpartial-inlining -fpeephole2 -freorder-functions \
           -frerun-cse-after-loop -fsched-interblock \
           -fstore-merging -fstrict-aliasing \
           -ftree-builtin-call-dce -ftree-switch-conversion -ftree-tail-merge \
           -fcode-hoisting -ftree-vrp -fipa-ra
CFLAGS = -mavx512f -mavx2 -mfma -O $(O2_REMOVE_REORDER) -std=c++17 -funroll-loops -fopenmp -fpermissive -mavx512vl -mavx512bw
#CFLAGS = -mavx512f -mavx2 -mfma -O2 -std=c++17 -fopenmp

# all: check lib test benchmark
all: check benchmark
check:
	@echo "CXX: " $(CXX)
	@echo "CXX VERSION: " $(shell expr `g++ -dumpversion`)
ifeq "$(GCCVERSIONGTEQ7)" "0"
	@echo "ERROR: GCC version MUST >= 7 to support 'constexpr if', according to https://en.cppreference.com/w/cpp/compiler_support#cpp17"
	exit
endif


benchmark: benchmark_packb


benchmark_nopack: benchmark_nopack.cpp gemm_kernel.h 
	$(CXX) $(CFLAGS) benchmark_nopack.cpp  -o benchmark_nopack -lmkl_rt -lxsmm -I/home2/xiaolany/libxsmm/include -L/home2/xiaolany/libxsmm/lib

benchmark_packb: benchmark_packb.cpp gemm_kernel.h test.h
	$(CXX) $(CFLAGS) benchmark_packb.cpp  -o benchmark_packb -lmkl_rt  -lxsmm -I/home2/xiaolany/libxsmm/include -L/home2/xiaolany/libxsmm/lib

clean:
	rm -rf benchmark_nopack  benchmark_packb 

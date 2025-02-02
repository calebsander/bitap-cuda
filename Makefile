PATTERN_LENGTH = 32

CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -DPATTERN_LENGTH=$(PATTERN_LENGTH)
LDFLAGS = -lcudart
NVCC_TARGET = -gencode arch=compute_52,code=sm_52

EXECUTABLES = agrep fgrep bench_exact_global bench_exact_reduce \
	bench_exact_shared bench_exact_warp_reduce bench_fuzzy_reduce

all: $(EXECUTABLES)

%.o: %.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -c $< -o $@

agrep: agrep.o bitap-cpu/pattern.o fuzzy_reduce.o
	$(CC) $^ $(LDFLAGS) -o $@
fgrep: fgrep.o bitap-cpu/pattern.o exact_reduce.o
	$(CC) $^ $(LDFLAGS) -o $@
bench_exact_global: bench_exact.c bitap-cpu/bench.c bitap-cpu/pattern.c exact_global.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -DBENCH $^ -lm -o $@
bench_exact_reduce: bench_exact.c bitap-cpu/bench.c bitap-cpu/pattern.c exact_reduce.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -DBENCH $^ -lm -o $@
bench_exact_shared: bench_exact.c bitap-cpu/bench.c bitap-cpu/pattern.c exact_shared.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -DBENCH $^ -lm -o $@
bench_exact_warp_reduce: bench_exact.c bitap-cpu/bench.c bitap-cpu/pattern.c exact_warp_reduce.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -DBENCH $^ -lm -o $@
bench_fuzzy_reduce: bench_fuzzy.c bitap-cpu/bench.c bitap-cpu/pattern.c fuzzy_reduce.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -DBENCH $^ -lm -o $@

agrep.o: bitap-cpu/fgrep.c bitap-cpu/fuzzy.h bitap-cpu/pattern.h cuda_utils.h
	$(CC) $(CFLAGS) -DCUDA -DFUZZY -c $< -o $@
fgrep.o: bitap-cpu/fgrep.c bitap-cpu/exact.h bitap-cpu/pattern.h cuda_utils.h
	$(CC) $(CFLAGS) -DCUDA -c $< -o $@
bitap-cpu/pattern.o: bitap-cpu/pattern.h
exact_global.o: bitap-cpu/exact.h bitap-cpu/bench.h bitap-cpu/pattern.h cuda_utils.h
exact_reduce.o: bitap-cpu/exact.h bitap-cpu/bench.h bitap-cpu/pattern.h cuda_utils.h
exact_shared.o: bitap-cpu/exact.h bitap-cpu/bench.h bitap-cpu/pattern.h cuda_utils.h
exact_warp_reduce.o: bitap-cpu/exact.h bitap-cpu/bench.h bitap-cpu/pattern.h cuda_utils.h
fuzzy_reduce.o: bitap-cpu/fuzzy.h bitap-cpu/bench.h bitap-cpu/pattern.h cuda_utils.h

clean:
	rm -f *.o $(EXECUTABLES)

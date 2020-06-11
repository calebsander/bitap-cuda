PATTERN_LENGTH = 32

CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -DPATTERN_LENGTH=$(PATTERN_LENGTH)
LDFLAGS = -lcudart
NVCC_TARGET = -gencode arch=compute_52,code=sm_52

EXECUTABLES = fgrep bench-exact-global bench-exact-reduce bench-exact-shared bench-exact-warp-reduce

all: $(EXECUTABLES)

%.o: %.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -c $< -o $@

fgrep: bitap-cpu/fgrep.o bitap-cpu/pattern.o exact_warp_reduce.o
	$(CC) $^ $(LDFLAGS) -o $@
bench-exact-global: bench-exact.c bitap-cpu/bench.c bitap-cpu/pattern.c exact_global.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -DBENCH $^ -lm -o $@
bench-exact-reduce: bench-exact.c bitap-cpu/bench.c bitap-cpu/pattern.c exact_reduce.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -DBENCH $^ -lm -o $@
bench-exact-shared: bench-exact.c bitap-cpu/bench.c bitap-cpu/pattern.c exact_shared.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -DBENCH $^ -lm -o $@
bench-exact-warp-reduce: bench-exact.c bitap-cpu/bench.c bitap-cpu/pattern.c exact_warp_reduce.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -DBENCH $^ -lm -o $@

bitap-cpu/fgrep.o: bitap-cpu/fgrep.c bitap-cpu/exact.h bitap-cpu/pattern.h cuda_utils.h
	$(CC) $(CFLAGS) -DCUDA -c $< -o $@
bitap-cpu/pattern.o: bitap-cpu/pattern.h
exact_global.o: exact.h bitap-cpu/bench.h cuda_utils.h
exact_reduce.o: exact.h bitap-cpu/bench.h cuda_utils.h
exact_shared.o: exact.h bitap-cpu/bench.h cuda_utils.h
exact_warp_reduce.o: exact.h bitap-cpu/bench.h cuda_utils.h

clean:
	rm -f *.o $(EXECUTABLES)

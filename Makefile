PATTERN_LENGTH = 32

CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -DPATTERN_LENGTH=$(PATTERN_LENGTH)
LDFLAGS = -lcudart
NVCC_TARGET = -gencode arch=compute_52,code=sm_52

EXECUTABLES = fgrep bench-exact-global

all: $(EXECUTABLES)

%.o: %.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -c $< -o $@

fgrep: bitap-cpu/fgrep.o bitap-cpu/pattern.o exact_global.o
	$(CC) $^ $(LDFLAGS) -o $@
bench-exact-global: bench-exact-global.c bitap-cpu/bench.c bitap-cpu/pattern.c exact_global.cu
	$(NVCC) -Xcompiler "$(CFLAGS)" -DBENCH $^ -lm -o $@

bitap-cpu/fgrep.o: bitap-cpu/fgrep.c bitap-cpu/exact.h bitap-cpu/pattern.h cuda_utils.h
	$(CC) $(CFLAGS) -DCUDA -c $< -o $@
bitap-cpu/pattern.o: bitap-cpu/pattern.h
exact_global.o: exact_global.h cuda_utils.h

clean:
	rm -f *.o $(EXECUTABLES)

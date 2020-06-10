PATTERN_LENGTH = 32

CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -DPATTERN_LENGTH=$(PATTERN_LENGTH)
LDFLAGS = -lcudart
NVCC_TARGET = -gencode arch=compute_52,code=sm_52

EXECUTABLES = fgrep

all: $(EXECUTABLES)

%.o: %.cu
	$(NVCC) $(NVCC_TARGET) -Xcompiler "$(CFLAGS)" -c $< -o $@

fgrep: bitap-cpu/fgrep.o exact_global.o bitap-cpu/pattern.o
	$(CC) $^ $(LDFLAGS) -o $@

bitap-cpu/fgrep.o: bitap-cpu/fgrep.c bitap-cpu/exact.h bitap-cpu/pattern.h cuda_utils.h
	$(CC) $(CFLAGS) -DCUDA -c $< -o $@
bitap-cpu/pattern.o: bitap-cpu/pattern.h
exact_global.o: exact_global.h cuda_utils.h

clean:
	rm -f *.o $(EXECUTABLES)

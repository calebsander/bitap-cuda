#ifndef UTILS_H
#define UTILS_H

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Asserts that a CUDA function call succeeds
#define CUDA_CALL(operation) do { \
	cudaError_t result = (operation); \
	if (result != cudaSuccess) { \
		fprintf(stderr, "Cuda failure: %s\n", cudaGetErrorString(result)); \
		assert(false); \
	} \
} while(0)

#ifdef __CUDACC__
	// Define atomicAdd() for size_t values as a 64-bit atomic add
	static_assert(sizeof(size_t) == sizeof(unsigned long long));

	__device__ __inline__ size_t atomicAdd(size_t *address, size_t value) {
		return atomicAdd((unsigned long long *) address, (unsigned long long) value);
	}
#endif

#endif // #ifndef UTILS_H

#include "exact_global.h"
#include <stdlib.h>
#include "bitap-cpu/bench.h"
#include "cuda_utils.h"

// Each thread applies the bitap algorithm to a contiguous section of the text.
// Since each thread's section overlaps with `pattern_length - 1`
// characters of the next section, high parallelism increases the accesses of the text.
#define THREADS_PER_BLOCK 1024
#define BLOCKS 32

__global__
void exact_kernel(
	const pattern_t *pattern,
	const unsigned char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
) {
	size_t pattern_length = pattern->length;
	unsigned grid_size = BLOCKS * THREADS_PER_BLOCK;
	unsigned thread_index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	// Each thread computes the `matches_mask` between `start_index` and `end_index`.
	// This finds any occurrence of the pattern
	// between `start_index` and the next thread's `start_index`.
	size_t start_index = text_length * thread_index / grid_size,
	       end_index = text_length * (thread_index + 1) / grid_size + pattern_length - 1;
	if (end_index > text_length) end_index = text_length;
	pattern_mask_t matches_mask = 0;
	pattern_mask_t end_mask = pattern->end_mask;
	for (size_t index = start_index; index < end_index; index++) {
		matches_mask = (matches_mask << 1 | 1) & pattern->char_masks[text[index]];
		if (matches_mask & end_mask) {
			// Threads must atomically reserve a slot in the `match_indices` array
			match_indices[atomicAdd(match_count, 1)] = index - pattern_length + 1;
		}
	}
}

int index_cmp(const void *_a, const void *_b) {
	size_t a = *(size_t *) _a, b = *(size_t *) _b;
	return a < b ? -1 : a > b ? +1 : 0;
}

void find_exact(
	const pattern_t *pattern,
	const char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
) {
	// Allocate memory for the variables on the GPU
	start_time(COPY_TO_GPU);
	pattern_t *dev_pattern;
	CUDA_CALL(cudaMalloc(&dev_pattern, sizeof(*dev_pattern)));
	CUDA_CALL(cudaMemcpy(dev_pattern, pattern, sizeof(*pattern), cudaMemcpyHostToDevice));
	unsigned char *dev_text;
	CUDA_CALL(cudaMalloc(&dev_text, sizeof(char[text_length])));
	CUDA_CALL(cudaMemcpy(
		dev_text,
		text,
		sizeof(char[text_length]),
		cudaMemcpyHostToDevice
	));
	size_t *dev_match_indices;
	CUDA_CALL(cudaMalloc(&dev_match_indices, sizeof(size_t[text_length])));
	size_t *dev_match_count;
	CUDA_CALL(cudaMalloc(&dev_match_count, sizeof(*dev_match_count)));
	CUDA_CALL(cudaMemset(dev_match_count, 0, sizeof(*dev_match_count)));
	stop_time();

	// Run the kernel
	start_time(FIND_MATCHES);
	exact_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
		dev_pattern,
		dev_text,
		text_length,
		dev_match_indices,
		dev_match_count
	);
	stop_time();

	// Free memory and copy outputs back to CPU
	start_time(COPY_FROM_GPU);
	CUDA_CALL(cudaFree(dev_pattern));
	CUDA_CALL(cudaFree(dev_text));
	CUDA_CALL(cudaMemcpy(
		match_count,
		dev_match_count,
		sizeof(*match_count),
		cudaMemcpyDeviceToHost
	));
	CUDA_CALL(cudaFree(dev_match_count));
	CUDA_CALL(cudaMemcpy(
		match_indices,
		dev_match_indices,
		sizeof(size_t[*match_count]),
		cudaMemcpyDeviceToHost
	));
	CUDA_CALL(cudaFree(dev_match_indices));
	stop_time();

	// Sort the match indices, since they can be returned out of order
	start_time(SORT_MATCHES);
	qsort(match_indices, *match_count, sizeof(*match_indices), index_cmp);
	stop_time();
}

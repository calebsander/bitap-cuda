#include "bitap-cpu/exact.h"
#include <stdlib.h>
#include "bitap-cpu/bench.h"
#include "cuda_utils.h"

#define THREADS_PER_BLOCK 1024
#define BLOCKS 8

#define div_ceil(a, b) (((a) + (b) - 1) / (b))

// 16 bytes
typedef ulonglong4 chars_t;

__global__
void exact_kernel(
	const pattern_t *pattern,
	const unsigned char *text,
	size_t text_length,
	size_t block_length,
	size_t *match_indices,
	size_t *match_count
) {
	extern __shared__ unsigned char block_text[];

	size_t block_start = text_length * blockIdx.x / BLOCKS;
	block_start &= ~(alignof(chars_t) - 1); // ensure accesses are aligned

	// Copy all characters that will be accessed by the block into shared memory.
	// Read 16 bytes at a time to improve transfer speeds
	for (
		size_t index = threadIdx.x * sizeof(chars_t);
		index < block_length;
		index += THREADS_PER_BLOCK * sizeof(chars_t)
	) {
		*(chars_t *) &block_text[index] = *(chars_t *) &text[block_start + index];
	}

	__syncthreads();

	size_t pattern_last_index = pattern->length - 1;
	unsigned grid_size = BLOCKS * THREADS_PER_BLOCK;
	unsigned thread_index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	size_t start_index = text_length * thread_index / grid_size,
	       end_index = text_length * (thread_index + 1) / grid_size + pattern_last_index;
	pattern_mask_t matches_mask = 0;
	pattern_mask_t end_mask = pattern->end_mask;
	for (size_t index = start_index; index < end_index; index++) {
		matches_mask = (matches_mask << 1 | 1) &
			pattern->char_masks[block_text[index - block_start]];
		if (matches_mask & end_mask) {
			match_indices[atomicAdd(match_count, 1)] = index - pattern_last_index;
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
	// Allocate a bit of extra memory to avoid kernel accessing past the end of `dev_text`
	CUDA_CALL(cudaMalloc(&dev_text, sizeof(char[
		text_length + pattern->length - 1 + alignof(chars_t) - 1
	])));
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

	// Compute the maximum number of characters that will be stored in shared memory.
	// This is the slice of text spanned by a block, plus the length of the pattern
	// (which overlaps with the next block), plus extra bytes for alignment.
	size_t block_length =
		div_ceil(text_length, BLOCKS) + pattern->length - 1 + alignof(chars_t) - 1;
	// Run the kernel
	start_time(FIND_MATCHES);
	exact_kernel<<<BLOCKS, THREADS_PER_BLOCK, sizeof(char[block_length])>>>(
		dev_pattern,
		dev_text,
		text_length,
		block_length,
		dev_match_indices,
		dev_match_count
	);
	CUDA_CALL(cudaDeviceSynchronize());
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

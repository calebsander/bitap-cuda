#include "exact.h"
#include <stdlib.h>
#include "bitap-cpu/bench.h"
#include "cuda_utils.h"

/**
 * This version is identical to the prefix-sum reduction version,
 * except it performs the reduction through warp shuffles.
 * This takes advantage of the fact that the pattern has the same length as a warp,
 * so it will only work when `PATTERN_LENGTH == 32`.
 * Using warp shuffles allows values to be transferred between registers,
 * reduces the number of __syncthreads() calls, and
 * results in a kernel with a smaller shared-memory footprint.
 */

#define THREADS_PER_BLOCK 1024
#define BLOCKS 8
#define BLOCK_STRIDE (THREADS_PER_BLOCK - PATTERN_LENGTH)
#define WARP_SIZE 32
#define WARP_ALL ((unsigned) -1)

static_assert(PATTERN_LENGTH == WARP_SIZE);
static_assert(THREADS_PER_BLOCK % WARP_SIZE == 0);

/** Performs the prefix-sum reduction using warp shuffles */
__device__ __inline__
pattern_mask_t prefix_non_matches_mask(pattern_mask_t mask, unsigned thread_index) {
	// Up-sweep
	#pragma unroll
	for (unsigned offset = 1; offset < WARP_SIZE; offset <<= 1) {
		pattern_mask_t received_mask = __shfl_up_sync(WARP_ALL, mask, offset);
		if (!(~thread_index & ((offset << 1) - 1))) mask |= received_mask << offset;
	}

	// Down-sweep
	#pragma unroll
	for (unsigned offset = WARP_SIZE >> 2; offset; offset >>= 1) {
		pattern_mask_t received_mask = __shfl_up_sync(WARP_ALL, mask, offset);
		unsigned sending_index = thread_index - offset;
		if (!(~sending_index & ((offset << 1) - 1)) && (~sending_index & (WARP_SIZE - 1))) {
			mask |= received_mask << offset;
		}
	}
	return mask;
}

__global__
void exact_kernel(
	const pattern_t *pattern,
	const unsigned char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
) {
	__shared__ pattern_mask_t warp_masks[THREADS_PER_BLOCK / WARP_SIZE];

	pattern_mask_t end_mask = pattern->end_mask;
	// Loop to cover the entire text with the thread grid
	for (
		size_t block_start = blockIdx.x * BLOCK_STRIDE;
		block_start < text_length;
		block_start += BLOCK_STRIDE * BLOCKS
	) {
		// Each thread computes the mask based on its current character
		unsigned thread_index = threadIdx.x;
		size_t text_index = block_start + thread_index;
		pattern_mask_t not_matches_mask = text_index < text_length
			? ~pattern->char_masks[text[text_index]]
			: ~(pattern_mask_t) 0; // if past the end of the text, no match is possible

		// Perform the warp-shuffle reduction
		not_matches_mask = prefix_non_matches_mask(not_matches_mask, thread_index);

		// We do need to send the last mask from this warp to the next warp,
		// which is done using shared memory
		unsigned warp = thread_index / WARP_SIZE;
		unsigned next_warp_index = (thread_index + 1) & (WARP_SIZE - 1);
		if (!next_warp_index) warp_masks[warp] = not_matches_mask;

		__syncthreads();

		if (next_warp_index) {
			// If the thread is not the last in its group,
			// it combines the mask from the end of the previous group
			pattern_mask_t previous_group_mask = warp
				? warp_masks[warp - 1]
				: ~(pattern_mask_t) 0; // the first warp has no previous warp, so 0 out the mask
			not_matches_mask |= previous_group_mask << next_warp_index;
		}

		// A match occurs if the `end_mask` bit is a 1 (0 in the inverted mask)
		if (~not_matches_mask & end_mask) {
			match_indices[atomicAdd(match_count, 1)] = text_index - pattern->length + 1;
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

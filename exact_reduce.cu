#include "exact.h"
#include <stdlib.h>
#include "bitap-cpu/bench.h"
#include "cuda_utils.h"

/**
 * This version uses a prefix-sum reduction to combine the characters' masks.
 * This requires a few changes:
 * - Each thread handles only 1 character at a time. Rather than covering the entire text
 *   with the thread grid, the grid repeatedly moves with stride BLOCK_STRIDE * BLOCKS.
 * - `reduced_masks` stores pattern masks with their bits *inverted*,
 *   i.e. 0 indicates a match and 1 indicates a mismatch.
 *   This is convenient so we don't need to shift in 1s when combining the masks.
 * - Each thread calculates the pattern mask for its character, and then
 *   neighboring threads combine their masks using a modified prefix-sum reduction.
 *   This reduction is unusual because:
 *   - The combining operation (mask1 << offset | mask2) depends on `offset`,
 *     so it differs between levels of the reduction
 *   - It's local, since each mask only uses the last PATTERN_LENGTH characters.
 *     If PATTERN_LENGTH is 2 ** n, this requires n levels of a prefix-sum reduction,
 *     plus 1 additional step to incorporate the final mask from the previous group.
 *   - It's not a proper-prefix sum (i.e. the first value is the first mask, not 0)
 */

#define THREADS_PER_BLOCK 1024
#define BLOCKS 8
#define BLOCK_STRIDE (THREADS_PER_BLOCK - PATTERN_LENGTH)

__global__
void exact_kernel(
	const pattern_t *pattern,
	const unsigned char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
) {
	__shared__ pattern_mask_t reduced_masks[THREADS_PER_BLOCK];

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
		reduced_masks[thread_index] = text_index < text_length
			? ~pattern->char_masks[text[text_index]]
			: ~(pattern_mask_t) 0; // if past the end of the text, no match is possible

		__syncthreads();

		/* Up-sweep phase.
		 * For example, suppose we are reducing 8 masks:
		 * A B C D E F G H
		 * In the first level, we combine A into B, C into D, E into F, and G into H.
		 * In the second level, we combine B into D and F into H.
		 * In the third level, we combine D into H.
		 */
		#pragma unroll
		for (unsigned offset = 1; offset < PATTERN_LENGTH; offset <<= 1) {
			if (!(~thread_index & ((offset << 1) - 1))) {
				reduced_masks[thread_index] |= reduced_masks[thread_index - offset] << offset;
			}

			__syncthreads();
		}

		/* Down-sweep phase.
		 * Continuing the example above, we have only 2 levels:
		 * In the first level, we combine D into F.
		 * In the second level, we combine B into C, D into E, and F into G.
		 */
		// The index of the next index within the PATTERN_LENGTH group of threads.
		// It is 0 for the last thread in the group.
		// We combine this last mask into the next group below, so we skip it here.
		unsigned next_group_index = (thread_index + 1) & (PATTERN_LENGTH - 1);
		#pragma unroll
		for (unsigned offset = PATTERN_LENGTH >> 2; offset; offset >>= 1) {
			if (!(~thread_index & ((offset << 1) - 1)) && next_group_index) {
				reduced_masks[thread_index + offset] |= reduced_masks[thread_index] << offset;
			}

			__syncthreads();
		}

		// Each thread retrieves its final mask after the reduction
		pattern_mask_t not_matches_mask = reduced_masks[thread_index];
		if (next_group_index) {
			// If the thread is not the last in its group,
			// it combines the mask from the end of the previous group
			unsigned group_start = thread_index & ~(PATTERN_LENGTH - 1);
			pattern_mask_t previous_group_mask = group_start
				? reduced_masks[group_start - 1]
				: ~(pattern_mask_t) 0; // the first group has no previous group, so 0 out the mask
			not_matches_mask |= previous_group_mask << next_group_index;
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

#include "fuzzy.h"
#include <stdlib.h>
#include "bitap-cpu/bench.h"
#include "cuda_utils.h"

/**
 * The GPU implementation of the fuzzy matching algorithm.
 * It is based on `exact_reduce.cu`, using a prefix-sum reduction
 * to calculate the matches masks at each distance.
 * (This could be done with a warp-shuffle approach too,
 * but it is barely faster and makes longer patterns difficult to support.)
 * As in the other reductions, the matches masks are inverted to avoid shifting in 1s.
 *
 * It includes a few optimizations over the CPU implementation.
 * The main one is based on the observation that only the matches masks
 * at the last distance and the current distance need to be stored,
 * so there are 2 shared-memory arrays that store the masks at these two distances.
 */

#define THREADS_PER_BLOCK 1024
#define BLOCKS 8
#define BLOCK_STRIDE (THREADS_PER_BLOCK - PATTERN_LENGTH)

/**
 * Performs the prefix-sum reduction to compute the matches masks at the next level.
 * This consolidates the logic in the CPU implementation for the 0 and nonzero distances.
 *
 * @param masks the array of matches mask to fill (it is initially uninitialize)
 * @param char_mask the mask for this character, computed from the pattern
 * @param error_mask additional matches made possible by assuming an error in the text.
 *   (This is inverted, so a 0 bit indicates a successful match using an error.)
 *   For the 0 distance, all bits are 1 since no errors are possible.
 * @param initial_mask the value of the mask before the first character of the text.
 *   (This is ~0 << distance.) Combined with the matches masks in the first warp.
 * @param thread_index the index of the thread in the block (and therefore, `masks`)
 */
__device__ __inline__ void reduce_char_mask(
	pattern_mask_t *masks,
	pattern_mask_t char_mask,
	pattern_mask_t error_mask,
	pattern_mask_t initial_mask,
	unsigned thread_index
) {
	// Set the current thread's initial mask by combining
	// the possibilities of a matched character or an additional error
	masks[thread_index] = char_mask & error_mask;

	__syncthreads();

	// Up-sweep phase. This is basically identical to the exact reduction,
	// except we include `error_mask` to preserve the possibility of an error.
	// (This can't be done at the end, since it needs to propagate in the reduction.)
	#pragma unroll
	for (unsigned offset = 1; offset < PATTERN_LENGTH; offset <<= 1) {
		if (!(~thread_index & ((offset << 1) - 1))) {
			masks[thread_index] |= (masks[thread_index - offset] << offset) & error_mask;
		}

		__syncthreads();
	}

	// Down-sweep phase. This more closely resembles the warp-shuffle code,
	// since we must combine *into* `thread_index` because we don't know other `error_mask`s
	#pragma unroll
	for (unsigned offset = PATTERN_LENGTH >> 2; offset; offset >>= 1) {
		unsigned sending_index = thread_index - offset;
		if (
			!(~sending_index & ((offset << 1) - 1)) &&
			(~sending_index & (PATTERN_LENGTH - 1))
		) {
			masks[thread_index] |= (masks[sending_index] << offset) & error_mask;
		}

		__syncthreads();
	}

	unsigned next_group_index = (thread_index + 1) & (PATTERN_LENGTH - 1);
	if (next_group_index) {
		// If the thread is not the last in its group,
		// it combines the mask from the end of the previous group
		unsigned group_start = thread_index & ~(PATTERN_LENGTH - 1);
		pattern_mask_t previous_group_mask = group_start
			? masks[group_start - 1]
			: initial_mask; // the first group has no previous group, so use the initial mask
		masks[thread_index] |= previous_group_mask << next_group_index;
	}
}

__global__
void fuzzy_kernel(
	const pattern_t *pattern,
	uint32_t errors,
	const unsigned char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
) {
	// The matches masks at the last error distance
	__shared__ pattern_mask_t fewer_error_masks[THREADS_PER_BLOCK];
	// The matches masks at the next error distance
	__shared__ pattern_mask_t next_error_masks[THREADS_PER_BLOCK];

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
		pattern_mask_t char_mask = text_index < text_length
			? ~pattern->char_masks[text[text_index]]
			: ~(pattern_mask_t) 0; // if past the end of the text, no match is possible

		pattern_mask_t initial_mask = ~(pattern_mask_t) 0;
		reduce_char_mask(
			fewer_error_masks,
			char_mask,
			~(pattern_mask_t) 0, // no possibility for errors at distance 0
			initial_mask,
			thread_index
		);

		for (uint32_t distance = errors; distance; distance--) {
			__syncthreads();

			// Compute the matches mask for each additional error possibility.
			// This uses the masks from the *previous* distance, so a reduction is unnecessary.
			// The `fewer_error_mask` at the previous index only exists for `thread_index > 0`.
			pattern_mask_t last_fewer_errors_mask = thread_index
				? fewer_error_masks[thread_index - 1]
				: ~(pattern_mask_t) 0;
			pattern_mask_t error_mask =
				// `last_fewer_errors_mask << 1`: a substitution at this character
				// `fewer_error_masks[thread_index] << 1`: a deletion after this character
				(last_fewer_errors_mask & fewer_error_masks[thread_index]) << 1 &
				// `last_fewer_errors_mask`: this character in the text was inserted
				last_fewer_errors_mask;

			// Compute the matches masks, reducing the `char_mask`s into `error_mask`
			initial_mask <<= 1;
			reduce_char_mask(next_error_masks, char_mask, error_mask, initial_mask, thread_index);

			// The current distance becomes the next distance
			fewer_error_masks[thread_index] = next_error_masks[thread_index];
		}

		// A match occurs if the `end_mask` bit is a 1 (0 in the inverted mask)
		if (~fewer_error_masks[thread_index] & end_mask) {
			// We don't know where the match started (due to deletions or insertions),
			// so just output its end index (exclusive)
			match_indices[atomicAdd(match_count, 1)] = text_index + 1;
		}

		__syncthreads();
	}
}

int index_cmp(const void *_a, const void *_b) {
	size_t a = *(size_t *) _a, b = *(size_t *) _b;
	return a < b ? -1 : a > b ? +1 : 0;
}

void find_fuzzy(
	const pattern_t *pattern,
	uint32_t errors,
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
	fuzzy_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
		dev_pattern,
		errors,
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

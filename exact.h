#ifndef EXACT_H
#define EXACT_H

#include "bitap-cpu/pattern.h"

// This interface is identical to the CPU exact matcher,
// but internally launches a CUDA kernel
#ifdef __cplusplus
extern "C"
#endif
void find_exact(
	const pattern_t *pattern,
	const char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
);

#endif // #ifndef EXACT_H

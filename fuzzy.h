#ifndef FUZZY_H
#define FUZZY_H

#include "bitap-cpu/pattern.h"

// This interface is identical to the CPU fuzzy matcher,
// but internally launches a CUDA kernel
#ifdef __cplusplus
extern "C"
#endif
void find_fuzzy(
	const pattern_t *pattern,
	uint32_t errors,
	const char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
);

#endif // #ifndef FUZZY_H

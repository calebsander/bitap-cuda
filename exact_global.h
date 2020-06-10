#ifndef EXACT_GLOBAL_H
#define EXACT_GLOBAL_H

#include "bitap-cpu/pattern.h"

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

#endif // #ifndef EXACT_GLOBAL_H
